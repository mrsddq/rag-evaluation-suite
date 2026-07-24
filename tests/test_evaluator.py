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


def test_dataset_report_renders_html():
    report = evaluate_dataset([sample_case()])
    assert report["count"] == 1
    assert "RAG evaluation" in render_html(report)

