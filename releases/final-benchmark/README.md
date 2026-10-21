# Dataset

**Release:** `final-benchmark-20260926-v7` | **525 questions**

| Split | Questions | Options |
| --- | ---: | --- |
| `binary/` | 346 | Correct / Incorrect |
| `multiclass/` | 179 | 3, 4, or 5 |

Each split contains:

- `inputs.jsonl`: model-facing questions and evidence, one JSON object per line.
- `answers.json`: scoring labels, keyed by `question_id` in a JSON array.
- `metadata.json`: basic source information in a JSON array.

`manifest.json` records the release version and dataset counts.

See the [main README](../../README.md) for model evaluation and scoring.
