# RAG Evaluation Suite

An offline, reproducible evaluation toolkit for retrieval-augmented generation. Feed it JSONL and receive per-case results, macro averages, and an optional standalone HTML dashboard.

## Metrics

- **Context precision**: average precision of relevant contexts in ranked retrieval
- **Context recall**: fraction of known relevant contexts retrieved
- **Reciprocal rank**: inverse rank of the first relevant context
- **Faithfulness**: fraction of answer claims lexically supported by retrieved context
- **Answer relevance**: token F1 between question and answer
- **Answer correctness**: token F1 against a reference answer

Metrics are deterministic and run without sending evaluation data to an external model.

## Run the example

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

rag-eval examples/evaluation.jsonl \
  --output rag-report.json \
  --html rag-report.html
```

Each JSONL row uses this shape:

```json
{
  "id": "capital-1",
  "question": "What is the capital of France?",
  "answer": "Paris is the capital of France.",
  "retrieved_contexts": ["Paris is the capital and most populous city of France."],
  "relevant_contexts": ["Paris is the capital and most populous city of France."],
  "reference_answer": "The capital of France is Paris."
}
```

## Interpretation

These lexical metrics are fast regression signals, not complete measures of semantic quality. Track them over a stable benchmark, inspect case-level failures, and supplement them with human review or calibrated model-based judging when nuance matters.

```bash
pytest
ruff check .
docker build -t rag-eval .
```

MIT licensed.
