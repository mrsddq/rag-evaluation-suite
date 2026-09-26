import pytest

from rag_eval import EvaluationCase, evaluate_case, evaluate_dataset
from rag_eval.report import render_html


def sample_case():
    relevant = "Paris is the capital and most populous city of France."
    return EvaluationCase(
        id="capital-1",
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        retrieved_contexts=(relevant, "Berlin is the capital of Germany."),
        relevant_contexts=(relevant,),
        reference_answer="The capital of France is Paris.",
    )


def test_good_case_scores_retrieval_and_generation():
    result = evaluate_case(sample_case())
    assert result.context_precision == 1.0
    assert result.context_recall == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.faithfulness == 1.0
    assert result.answer_correctness > 0.9


def test_irrelevant_first_result_reduces_rank_metrics():
    case = sample_case()
    reordered = EvaluationCase(
        **{**case.__dict__, "retrieved_contexts": tuple(reversed(case.retrieved_contexts))}
    )
    result = evaluate_case(reordered)
    assert result.reciprocal_rank == 0.5
    assert result.context_precision == 0.5


@pytest.mark.parametrize(
    ("retrieved", "relevant", "precision", "recall", "reciprocal_rank"),
    [
        (("Paris France", "Paris France"), ("Paris France",), 1.0, 1.0, 1.0),
        (("Paris France", "France Paris"), ("Paris France",), 1.0, 1.0, 1.0),
        (("Berlin Germany", "Paris France", "Paris France"), ("Paris France",),
         0.5833, 1.0, 0.5),
        (("Paris France", "Berlin Germany"), ("Paris France", "Rome Italy"),
         1.0, 0.5, 1.0),
        ((), ("Paris France",), 0.0, 0.0, 0.0),
        (("Paris France",), (), 0.0, 0.0, 0.0),
        ((), (), 0.0, 0.0, 0.0),
        (("Berlin Germany",), ("Paris France",), 0.0, 0.0, 0.0),
    ],
    ids=[
        "duplicate-relevant-contexts",
        "distinct-contexts-matching-one-reference",
        "duplicates-preserve-rank-penalty",
        "recall-measures-missing-reference-coverage",
        "no-retrieved-contexts",
        "no-relevant-contexts",
        "both-context-lists-empty",
        "no-relevant-hits",
    ],
)
def test_ranked_context_precision(retrieved, relevant, precision, recall, reciprocal_rank):
    case = EvaluationCase(
        id="ranked-contexts",
        question="What is the capital of France?",
        answer="Paris.",
        retrieved_contexts=retrieved,
        relevant_contexts=relevant,
    )
    result = evaluate_case(case)
    assert result.context_precision == precision
    assert result.context_recall == recall
    assert result.reciprocal_rank == reciprocal_rank


def test_dataset_report_renders_html():
    report = evaluate_dataset([sample_case()])
    assert report["count"] == 1
    assert "RAG evaluation" in render_html(report)


@pytest.mark.parametrize("field,value", [
    ("retrieved_contexts", "Paris France"),
    ("relevant_contexts", {"context": "Paris France"}),
    ("retrieved_contexts", [None]),
    ("question", None),
    ("answer", 42),
    ("reference_answer", []),
])
def test_jsonl_rejects_wrong_field_types_with_line_number(tmp_path, field, value):
    import json
    from rag_eval.cli import load_jsonl

    row = {"question": "What is the capital?", "answer": "Paris", field: value}
    path = tmp_path / "cases.jsonl"
    path.write_text("\n" + json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_jsonl(path)

