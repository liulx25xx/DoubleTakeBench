#!/usr/bin/env python3
"""Generate offline baselines and score DoubleTakeBench predictions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "releases" / "final-benchmark"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def load_public_inputs(release: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = read_json(release / "manifest.json")
    items = {}
    for split in ("binary", "multiclass"):
        for packet in read_jsonl(release / split / "inputs.jsonl"):
            qid = packet["question_id"]
            items[qid] = {"question_id": qid, "split": split, "packet": packet}
    return manifest, items


def load_release(release: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest, items = load_public_inputs(release)
    for split in ("binary", "multiclass"):
        for answer in read_json(release / split / "answers.json"):
            items[answer["question_id"]]["target"] = answer["evaluation_target"]
    return manifest, items


def baseline_answer(item: dict[str, Any], kind: str, rng: random.Random) -> str:
    options = item["packet"]["options"]
    if kind == "first-option":
        return options[0]["id"]
    if kind == "random":
        return rng.choice(options)["id"]
    if kind == "longest-option":
        return max(options, key=lambda option: len(option["text"]))["id"]
    if kind == "shortest-option":
        return min(options, key=lambda option: len(option["text"]))["id"]
    return "INSUFFICIENT"


def make_baseline(items: dict[str, dict[str, Any]], kind: str, seed: int) -> list[dict[str, Any]]:
    return [
        {"question_id": qid,
         "answer": baseline_answer(item, kind, random.Random(f"{seed}:{qid}")),
         "status": "ok", "model": f"baseline:{kind}", "seed": seed}
        for qid, item in items.items()
    ]


def score_predictions(items: dict[str, dict[str, Any]], path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    counts = Counter(row["question_id"] for row in rows)
    predictions = {row["question_id"]: row for row in rows}

    def aggregate(qids: list[str]) -> dict[str, Any]:
        covered = correct = 0
        for qid in qids:
            row = predictions.get(qid, {})
            answer = row.get("answer")
            allowed = items[qid]["packet"]["response_contract"]["answer_ids"]
            if counts[qid] == 1 and row.get("status", "ok") in ("ok", "success") \
                    and answer in allowed and answer != "DEFER":
                covered += 1
                correct += answer == items[qid]["target"]
        total = len(qids)
        return {"n": total, "covered": covered, "correct": correct,
                "accuracy": correct / total if total else None,
                "coverage": covered / total if total else None,
                "accuracy_on_covered": correct / covered if covered else None}

    groups = {"all": list(items)}
    for split in ("binary", "multiclass"):
        groups[split] = [qid for qid, item in items.items() if item["split"] == split]
    for count in sorted({len(item["packet"]["options"]) for item in items.values()}):
        groups[f"{count}-option"] = [
            qid for qid, item in items.items() if len(item["packet"]["options"]) == count
        ]
    return {"questions": len(items),
            "groups": {name: aggregate(qids) for name, qids in groups.items()}}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline", help="generate and score an offline baseline")
    baseline.add_argument("--kind", choices=("first-option", "random", "longest-option",
                                           "shortest-option", "always-insufficient"),
                          default="random")
    baseline.add_argument("--seed", type=int, default=7)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--score-output", type=Path)
    score = sub.add_parser("score", help="score model predictions")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest, items = load_release(args.release)
    if args.command == "baseline":
        write_jsonl(args.output, make_baseline(items, args.kind, args.seed))
        result = score_predictions(items, args.output)
        result["baseline"] = {"kind": args.kind, "seed": args.seed}
        output = args.score_output
    else:
        result = score_predictions(items, args.predictions)
        output = args.output
    result["release"] = manifest["version"]
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
