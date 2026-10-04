from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    try:
        import numpy as np

        embeddings = _semantic_encoder().encode(
            sentences, normalize_embeddings=True, show_progress_bar=False
        )
        similarities = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    except Exception as exc:
        # A deterministic lexical fallback keeps local/offline execution usable.
        print(f"  ⚠️  Semantic model unavailable, using lexical fallback: {exc}")
        similarities = [_token_similarity(a, b) for a, b in zip(sentences, sentences[1:])]

    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if float(similarity) < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


@lru_cache(maxsize=1)
def _semantic_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.lower()))
    right_tokens = set(re.findall(r"\w+", right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _pack_text(text: str, max_size: int) -> list[str]:
    """Pack paragraphs/words without silently dropping oversized input."""
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    pieces: list[str] = []
    for paragraph in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not paragraph:
            continue
        if len(paragraph) <= max_size:
            pieces.append(paragraph)
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            if len(word) > max_size:
                if current:
                    pieces.append(current)
                    current = ""
                pieces.extend(word[i:i + max_size] for i in range(0, len(word), max_size))
            elif not current:
                current = word
            elif len(current) + 1 + len(word) <= max_size:
                current += " " + word
            else:
                pieces.append(current)
                current = word
        if current:
            pieces.append(current)

    packed: list[str] = []
    current = ""
    for piece in pieces:
        separator = "\n\n" if current else ""
        if current and len(current) + len(separator) + len(piece) > max_size:
            packed.append(current)
            current = piece
        else:
            current += separator + piece
    if current:
        packed.append(current)
    return packed


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = re.sub(r"[^\w.-]+", "_", str(metadata.get("source", "document")))

    for parent_index, parent_text in enumerate(_pack_text(text, parent_size)):
        parent_id = f"{source}:parent_{parent_index}"
        parent = Chunk(
            text=parent_text,
            metadata={
                **metadata,
                "chunk_type": "parent",
                "parent_id": parent_id,
                "chunk_index": parent_index,
            },
        )
        parents.append(parent)
        for child_index, child_text in enumerate(_pack_text(parent_text, child_size)):
            children.append(
                Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                        "chunk_index": child_index,
                    },
                    parent_id=parent_id,
                )
            )
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    header_pattern = re.compile(r"^#{1,3}\s+.+$", re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    chunks: list[Chunk] = []

    if not matches:
        stripped = text.strip()
        return ([Chunk(stripped, {**metadata, "section": "", "strategy": "structure"})]
                if stripped else [])

    preamble = text[:matches[0].start()].strip()
    if preamble:
        chunks.append(Chunk(preamble, {**metadata, "section": "preamble", "strategy": "structure"}))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[match.start():end].strip()
        header = match.group(0).lstrip("#").strip()
        if section_text:
            chunks.append(
                Chunk(
                    section_text,
                    {**metadata, "section": header, "strategy": "structure", "chunk_index": len(chunks)},
                )
            )
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
