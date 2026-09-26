from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from statistics import mean


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
    "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
}


def tokens(text: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(text) if word.lower() not in STOP_WORDS}


def token_f1(expected: str, actual: str) -> float:
    expected_tokens, actual_tokens = tokens(expected), tokens(actual)
    if not expected_tokens or not actual_tokens:
        return 0.0
    shared = len(expected_tokens & actual_tokens)
    precision = shared / len(actual_tokens)
    recall = shared / len(expected_tokens)
    return 2 * precision * recall / (precision + recall) if shared else 0.0


def is_relevant(retrieved: str, relevant_contexts: tuple[str, ...]) -> bool:
    retrieved_tokens = tokens(retrieved)
    for relevant in relevant_contexts:
        relevant_tokens = tokens(relevant)
        union = retrieved_tokens | relevant_tokens
        if retrieved.strip() == relevant.strip() or (union and len(retrieved_tokens & relevant_tokens) / len(union) >= 0.55):
            return True
    return False


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    answer: str
    retrieved_contexts: tuple[str, ...]
    relevant_contexts: tuple[str, ...] = ()
    reference_answer: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "EvaluationCase":
        if not isinstance(value, dict):
            raise TypeError("Each evaluation case must be an object")
        for field in ("question", "answer", "reference_answer"):
            if field in value and not isinstance(value[field], str):
                raise TypeError(f"{field} must be a string")
        for field in ("retrieved_contexts", "relevant_contexts"):
            contexts = value.get(field, [])
            if (not isinstance(contexts, (list, tuple))
                    or any(not isinstance(context, str) for context in contexts)):
                raise TypeError(f"{field} must be an array of strings")
        return cls(
            id=str(value.get("id", "")),
            question=value["question"],
            answer=value["answer"],
            retrieved_contexts=tuple(value.get("retrieved_contexts", [])),
            relevant_contexts=tuple(value.get("relevant_contexts", [])),
            reference_answer=value.get("reference_answer", ""),
        )


@dataclass(frozen=True)
class EvaluationResult:
    id: str
    context_precision: float
    context_recall: float
    reciprocal_rank: float
    faithfulness: float
    answer_relevance: float
    answer_correctness: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _retrieval_metrics(case: EvaluationCase) -> tuple[float, float, float]:
    if not case.relevant_contexts:
        return 0.0, 0.0, 0.0
    labels = [is_relevant(context, case.relevant_contexts) for context in case.retrieved_contexts]
    precisions: list[float] = []
    hits = 0
    for rank, relevant in enumerate(labels, 1):
        if relevant:
            hits += 1
            precisions.append(hits / rank)
    # Average over relevant retrieved ranks, since multiple contexts can match one reference.
    precision = mean(precisions) if precisions else 0.0

    found = sum(any(is_relevant(relevant, (item,)) for item in case.retrieved_contexts) for relevant in case.relevant_contexts)
    recall = found / len(case.relevant_contexts)
    reciprocal_rank = next((1 / rank for rank, relevant in enumerate(labels, 1) if relevant), 0.0)
    return precision, recall, reciprocal_rank


def _faithfulness(answer: str, contexts: tuple[str, ...]) -> float:
    claims = [claim for claim in SENTENCE_RE.split(answer) if tokens(claim)]
    if not claims or not contexts:
        return 0.0
    context_tokens = tokens(" ".join(contexts))
    supported = 0
    for claim in claims:
        claim_tokens = tokens(claim)
        if claim_tokens and len(claim_tokens & context_tokens) / len(claim_tokens) >= 0.6:
            supported += 1
    return supported / len(claims)


def evaluate_case(case: EvaluationCase) -> EvaluationResult:
    precision, recall, reciprocal_rank = _retrieval_metrics(case)
    correctness = token_f1(case.reference_answer, case.answer) if case.reference_answer else 0.0
    return EvaluationResult(
        id=case.id,
        context_precision=round(precision, 4),
        context_recall=round(recall, 4),
        reciprocal_rank=round(reciprocal_rank, 4),
        faithfulness=round(_faithfulness(case.answer, case.retrieved_contexts), 4),
        answer_relevance=round(token_f1(case.question, case.answer), 4),
        answer_correctness=round(correctness, 4),
    )


def evaluate_dataset(cases: list[EvaluationCase]) -> dict[str, object]:
    if not cases:
        raise ValueError("Dataset contains no evaluation cases")
    results = [evaluate_case(case) for case in cases]
    metric_names = (
        "context_precision", "context_recall", "reciprocal_rank",
        "faithfulness", "answer_relevance", "answer_correctness",
    )
    summary = {
        name: round(mean(getattr(result, name) for result in results), 4)
        for name in metric_names
    }
    return {"summary": summary, "cases": [result.to_dict() for result in results], "count": len(results)}

