from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("RAGAS inputs must have the same length")

    try:
        from config import OPENAI_API_KEY
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        frame = result.to_pandas()
        per_question = [
            EvalResult(
                question=str(row["question"]),
                answer=str(row["answer"]),
                contexts=list(row["contexts"]),
                ground_truth=str(row["ground_truth"]),
                faithfulness=_safe_float(row.get("faithfulness", 0.0)),
                answer_relevancy=_safe_float(row.get("answer_relevancy", 0.0)),
                context_precision=_safe_float(row.get("context_precision", 0.0)),
                context_recall=_safe_float(row.get("context_recall", 0.0)),
            )
            for _, row in frame.iterrows()
        ]
        aggregates = {
            name: (sum(getattr(item, name) for item in per_question) / len(per_question)
                   if per_question else 0.0)
            for name in metric_names
        }
        return {**aggregates, "per_question": per_question}
    except Exception as exc:
        print(f"  ⚠️  RAGAS evaluation failed: {exc}")
        fallback_results = [
            EvalResult(question, answer, list(context), ground_truth, 0.0, 0.0, 0.0, 0.0)
            for question, answer, context, ground_truth
            in zip(questions, answers, contexts, ground_truths)
        ]
        return {**{name: 0.0 for name in metric_names}, "per_question": fallback_results}


def _safe_float(value) -> float:
    try:
        number = float(value)
        return 0.0 if math.isnan(number) or math.isinf(number) else number
    except (TypeError, ValueError):
        return 0.0


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời chứa thông tin không được hỗ trợ bởi context",
            "Siết prompt chỉ dùng context, giảm temperature và kiểm tra citation",
        ),
        "context_recall": (
            "Retriever bỏ sót chunk liên quan",
            "Điều chỉnh chunking, tăng candidate pool hoặc bổ sung BM25/metadata filter",
        ),
        "context_precision": (
            "Context có quá nhiều chunk không liên quan",
            "Cải thiện hybrid weights và dùng cross-encoder reranking",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa trực tiếp giải quyết câu hỏi",
            "Cải thiện answer prompt và yêu cầu trả lời ngắn, đúng trọng tâm",
        ),
    }
    analyzed = []
    for result in eval_results:
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "expected": result.ground_truth,
            "answer": result.answer,
            "contexts": result.contexts,
            "worst_metric": worst_metric,
            "score": float(scores[worst_metric]),
            "average_score": sum(scores.values()) / len(scores),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    analyzed.sort(key=lambda item: item["average_score"])
    return analyzed[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
