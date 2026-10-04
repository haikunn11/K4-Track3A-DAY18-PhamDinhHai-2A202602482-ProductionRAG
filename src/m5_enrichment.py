from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize failed: {exc}")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Mỗi câu một dòng."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            questions = [re.sub(r"^\s*\d+[.)-]?\s*", "", q).strip() for q in content.splitlines()]
            return [q for q in questions if q][:n_questions]
        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA failed: {exc}")

    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    return [f"Thông tin nào được nêu về: {sentence.rstrip('.')}?" for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = ""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            response = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết đúng 1 câu ngắn mô tả vị trí và chủ đề của đoạn văn trong tài liệu."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0,
            )
            context = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual failed: {exc}")
    if not context and document_title:
        context = f"Trích từ tài liệu {document_title}."
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY:
        try:
            return _openai_json(
                'Trích xuất metadata và trả JSON: {"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}',
                text,
                max_tokens=150,
            )
        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata failed: {exc}")

    lowered = text.lower()
    category = "it" if any(word in lowered for word in ("mật khẩu", "vpn", "bảo mật")) else "policy"
    if any(word in lowered for word in ("lương", "chi phí", "vnđ", "tài chính")):
        category = "finance"
    elif any(word in lowered for word in ("nhân viên", "nghỉ phép", "thử việc")):
        category = "hr"
    first_line = next((line.strip("# ") for line in text.splitlines() if line.strip()), "general")
    return {"topic": first_line[:80], "entities": [], "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if OPENAI_API_KEY:
        try:
            return _openai_json(
                """Phân tích đoạn văn và trả về JSON gồm:
{"summary":"tóm tắt 2-3 câu","questions":["3 câu hỏi"],
"context":"1 câu mô tả vị trí/chủ đề",
"metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}}""",
                f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                max_tokens=400,
            )
        except Exception as exc:
            print(f"  ⚠️  Enrichment API failed: {exc}")

    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": f"Trích từ tài liệu {source}." if source else "",
        "metadata": extract_metadata(text),
    }


def _openai_json(system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = (response.choices[0].message.content or "{}").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE)
    parsed = json.loads(content)
    return parsed if isinstance(parsed, dict) else {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            source_line = f"Nguồn: {source}." if source else ""
            prefix = " ".join(part for part in (source_line, context_line) if part)
            enriched_text = f"{prefix}\n\n{text}" if prefix else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
