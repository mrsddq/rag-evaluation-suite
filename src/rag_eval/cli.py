from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluator import EvaluationCase, evaluate_dataset
from .report import render_html


def load_jsonl(path: Path) -> list[EvaluationCase]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid JSONL at line {line_number}: {exc}") from exc
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG outputs from a JSONL dataset")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", "-o", type=Path, default=Path("rag-report.json"))
    parser.add_argument("--html", type=Path, help="Also create a standalone HTML dashboard")
    args = parser.parse_args()
    report = evaluate_dataset(load_jsonl(args.dataset))
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.html:
        args.html.write_text(render_html(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

