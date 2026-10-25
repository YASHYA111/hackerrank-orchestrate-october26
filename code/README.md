# Ghost Data Still Exists solution

## Setup and run

From the repository root, install the code dependencies and set `OPENAI_API_KEY` in the process environment:

```powershell
python -m pip install -r code\requirements.txt
$env:OPENAI_API_KEY = "your-key"
python code\main.py
```

The key is never read from a file or written to logs. The default run reads participant-facing inputs from `dataset\`, processes exactly 518 requests, writes the root `output.csv`, and updates `code\evaluation\usage_report.md`. Use `--dataset` or `--output` to override paths, `--model` to change the OpenAI model, and `--no-images` to omit Vision inputs. Images are sent by default and can materially increase API cost.

## Pipeline

- `identity_resolver.py` joins current and historical email, phone, hash, and user-ID references, including identity links in activity evidence, at each case's `audit_at`.
- `evidence_aggregator.py` uses exact identifier matches to select evidence. The evidence manifest locates files; it does not map files to cases. SQL snapshot exports are executed in an in-memory SQLite database and only matching rows are put in the model context. File and event timestamps after `audit_at` are excluded.
- `perception_engine.py` uses GPT-4o Structured Outputs to extract intent, evidence-linked records, active flows, and directly supported compliant states. The model is an extractor; it does not make the final status or remediation decision.
- `deterministic_router.py` applies deletion-rule actions and builds a NetworkX DAG so active flows and restore blocks precede record changes. It emits only contract-approved phases, operations, and states.
- `main.py` validates the request count and identifiers, serializes output rows deterministically, and writes token and cost totals from API usage metadata.

Missing optional PDF text extraction is reported as a warning; install the declared `pypdf` dependency to include PDFs. Missing manifest files are also reported. Missing required inputs, an unavailable API key, invalid model output, or unsupported policy data stop the run instead of producing a success-shaped output.

## Tests

Run offline unit tests from the repository root:

```powershell
python -m unittest discover -s code -p "test_*.py"
```

The generated usage report estimates GPT-4o costs at $2.50 per million input tokens and $10.00 per million output tokens; provider pricing can change.
