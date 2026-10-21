#!/usr/bin/env python3
"""Run an OpenAI Responses-compatible model on DoubleTakeBench."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from public_benchmark import DEFAULT_RELEASE, load_public_inputs, read_jsonl, write_jsonl


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_OUTPUT_TOKENS = 32


def packet_prompt(packet: dict[str, Any]) -> str:
    lines = [
        "You are evaluating one coding-agent observation judgment.",
        "Use only the background, assumptions, evidence, and question below.",
        "Return exactly one answer ID from the listed answer choices.",
        "Do not return an explanation or any other text.",
        "",
        "BACKGROUND:",
        packet["background"],
        "",
        "STATE ASSUMPTIONS:",
    ]
    lines.extend(f"- {value}" for value in packet["state_assumptions"])
    lines.extend(["", "EVIDENCE:"])
    for evidence in packet["evidence"]:
        lines.append(f"[{evidence['id']}] {evidence['text']}")
    lines.extend(["", "QUESTION:", packet["question"], "", "ANSWER CHOICES:"])
    for option in packet["options"]:
        lines.append(f"{option['id']}: {option['text']}")
    lines.extend([
        "",
        "If the evidence does not support any listed factual option, return INSUFFICIENT.",
        "If the request cannot be answered from the supplied packet, return DEFER.",
        "FINAL ANSWER ID:",
    ])
    return "\n".join(lines)


def request_payload(model: str, prompt: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
    }


def request_json(url: str, api_key: str, payload: dict[str, Any], timeout: float,
                org_id: str | None = None) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if org_id:
        headers["OpenAI-Organization"] = org_id
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1_000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc
    return json.loads(raw)


def response_text(response: dict[str, Any]) -> str:
    """Extract text from Responses API and common Chat Completions shapes."""
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = response.get("output")
    chunks: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        chunks.append(part["text"])
    if chunks:
        return "".join(chunks)

    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
    raise RuntimeError("provider response contained no text output")


def _json_answer(text: str) -> str | None:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        candidates.append(fenced.group(1))
    object_match = re.search(r"\{[^{}]*\}", text, flags=re.S)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            for key in ("answer", "answer_id", "choice", "label", "prediction"):
                answer = value.get(key)
                if isinstance(answer, str) and answer.strip():
                    return answer.strip()
    return None


def parse_answer(text: str, allowed: set[str]) -> str | None:
    """Read an answer ID from plain text or JSON."""
    value = _json_answer(text)
    if value in allowed:
        return value
    stripped = text.strip().strip("`* \t\r\n")
    if stripped in allowed:
        return stripped
    pattern = r"^\s*(?:final answer|answer|choice|label)\s*(?:is|=|:)?\s*([A-Za-z0-9_-]+)\s*$"
    matches = re.findall(pattern, text, flags=re.I | re.M)
    if len(matches) == 1 and matches[0] in allowed:
        return matches[0]
    return None


def read_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["question_id"]: row for row in read_jsonl(path)}


def write_predictions(path: Path, predictions: dict[str, dict[str, Any]]) -> None:
    write_jsonl(path, list(predictions.values()))


def iter_items(items: dict[str, dict[str, Any]], split: str, limit: int | None):
    count = 0
    for item in items.values():
        if split != "all" and item["split"] != split:
            continue
        if limit is not None and count >= limit:
            break
        yield item
        count += 1


def run(args: argparse.Namespace) -> int:
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ["OPENAI_MODEL"]
    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    endpoint = base_url if base_url.endswith("/responses") else f"{base_url}/responses"
    org_id = os.environ.get("OPENAI_ORG_ID")

    manifest, items = load_public_inputs(args.release)
    existing = read_existing_predictions(args.output) if args.resume else {}
    if not args.resume:
        write_predictions(args.output, existing)
    for item in iter_items(items, args.split, args.limit):
        qid = item["question_id"]
        old_row = existing.get(qid)
        if old_row and old_row.get("status") == "ok" \
                and old_row.get("answer") in item["packet"]["response_contract"]["answer_ids"]:
            continue
        started = time.monotonic()
        row: dict[str, Any] = {
            "question_id": qid,
            "status": "error",
            "model": model,
            "release": manifest["version"],
            "max_output_tokens": args.max_output_tokens,
        }
        try:
            response = request_json(
                endpoint,
                api_key,
                request_payload(model, packet_prompt(item["packet"]), args.max_output_tokens),
                args.timeout,
                org_id,
            )
            raw = response_text(response)
            allowed = set(item["packet"]["response_contract"]["answer_ids"])
            answer = parse_answer(raw, allowed)
            row["status"] = "ok" if answer is not None else "invalid"
            if answer is not None:
                row["answer"] = answer
            row["raw_output"] = raw
        except TimeoutError:
            row["status"] = "timeout"
            row["error"] = "request timed out"
        except (RuntimeError, OSError, ValueError) as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        row["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
        existing[qid] = row
        write_predictions(args.output, existing)
        if args.delay:
            time.sleep(args.delay)
        print(json.dumps({"question_id": qid, "status": row["status"]}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--split", choices=("all", "binary", "multiclass"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--delay", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
