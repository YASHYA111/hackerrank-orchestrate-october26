from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAIError

from deterministic_router import DeterministicRouter
from evidence_aggregator import EvidenceAggregator, EvidenceBundle
from identity_resolver import TemporalIdentityResolver
from perception_engine import PerceptionEngine, PerceptionResult


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit deletion requests against supplied evidence.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "dataset",
        help="Directory containing participant-facing input files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "output.csv",
        help="Destination CSV path.",
    )
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model used for structured perception.")
    parser.add_argument("--no-images", action="store_true", help="Exclude image evidence from Vision requests.")
    return parser.parse_args()


def _validate_inputs(dataset_dir: Path) -> None:
    required = (
        "deletion_requests.csv",
        "user_identities.csv",
        "user_audit_log.csv",
        "system_activity.csv",
        "deletion_rules.csv",
        "evidence_files.csv",
    )
    missing = [name for name in required if not (dataset_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset input(s): {', '.join(missing)}")


def _write_output(output_path: Path, rows: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["case_id", "deletion_status", "steps_json"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "deletion_status": row["deletion_status"],
                    "steps_json": json.dumps(row["steps"], ensure_ascii=False, separators=(",", ":")),
                }
            )


def _write_usage_report(report_path: Path, model: str, usage: Any, request_count: int) -> None:
    total_tokens = usage.input_tokens + usage.output_tokens
    average = total_tokens / request_count if request_count else 0
    estimated_cost = usage.input_tokens * 2.5 / 1_000_000 + usage.output_tokens * 10 / 1_000_000
    per_request_cost = estimated_cost / request_count if request_count else 0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Model Usage Report",
                "",
                "Automatically generated from the completed run. GPT-4o cost uses an estimate of "
                "$2.50 per million input tokens and $10.00 per million output tokens; actual provider pricing may vary.",
                "",
                "| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Estimated cost |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                f"| OpenAI | {model} | {usage.calls} | {usage.input_tokens} | {usage.output_tokens} | "
                f"{total_tokens} | ${estimated_cost:.6f} |",
                "",
                f"- Requests processed: {request_count}",
                f"- Average total tokens per request: {average:.2f}",
                f"- Estimated cost per request: ${per_request_cost:.6f}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def _validate_evidence_references(perception: PerceptionResult, evidence: EvidenceBundle) -> None:
    available = {
        item["evidence_id"] for item in evidence.text_evidence
    } | {
        item["evidence_id"] for item in evidence.images
    } | {
        item["activity_id"] for item in evidence.activities
    }
    referenced = [
        item.evidence_id
        for item in (
            perception.records_found
            + perception.active_flows
            + perception.verified_states
        )
    ]
    unknown = sorted({evidence_id for evidence_id in referenced if evidence_id not in available})
    if unknown:
        raise ValueError(f"Perception returned evidence references not provided to the model: {', '.join(unknown)}")


def run(dataset_dir: Path, output_path: Path, model: str, include_images: bool = True) -> None:
    dataset_dir = dataset_dir.resolve()
    _validate_inputs(dataset_dir)
    requests = pd.read_csv(dataset_dir / "deletion_requests.csv", dtype=str, keep_default_na=False)
    expected_columns = {"case_id", "email", "audit_at", "request_text"}
    missing_columns = expected_columns - set(requests.columns)
    if missing_columns:
        raise ValueError(f"deletion_requests.csv is missing columns: {', '.join(sorted(missing_columns))}")
    if len(requests) != 518:
        raise ValueError(f"Expected 518 deletion requests, found {len(requests)}.")
    if requests["case_id"].duplicated().any():
        raise ValueError("deletion_requests.csv contains duplicate case_id values.")

    resolver = TemporalIdentityResolver(
        dataset_dir / "user_identities.csv",
        dataset_dir / "user_audit_log.csv",
        dataset_dir / "system_activity.csv",
    )
    aggregator = EvidenceAggregator(
        dataset_dir,
        dataset_dir / "evidence_files.csv",
        dataset_dir / "system_activity.csv",
        include_images=include_images,
    )
    policy_rules = pd.read_csv(
        dataset_dir / "deletion_rules.csv",
        dtype=str,
        keep_default_na=False,
    ).to_dict(orient="records")
    router = DeterministicRouter(str(dataset_dir / "deletion_rules.csv"))
    engine = PerceptionEngine(model=model)
    output_rows: list[dict[str, Any]] = []

    for request in requests.to_dict(orient="records"):
        identities = resolver.resolve(request["email"], request["audit_at"])
        evidence = aggregator.collect(identities, request["audit_at"])
        perception = engine.analyze(
            request["request_text"],
            identities,
            evidence,
            request["audit_at"],
            policy_rules,
        )
        if perception.intent_is_deletion:
            _validate_evidence_references(perception, evidence)
        decision = router.decide(perception)
        output_rows.append(
            {
                "case_id": request["case_id"],
                "deletion_status": decision.status,
                "steps": decision.steps,
            }
        )

    output_rows.sort(key=lambda row: row["case_id"])
    if len(output_rows) != len(requests) or len({row["case_id"] for row in output_rows}) != len(requests):
        raise ValueError("Output row count or case_id uniqueness does not match the input requests.")
    _write_output(output_path, output_rows)
    _write_usage_report(
        Path(__file__).resolve().parent / "evaluation" / "usage_report.md",
        model,
        engine.usage,
        len(output_rows),
    )
    print(f"Wrote {len(output_rows)} case results to {output_path}")


def main() -> int:
    args = _arguments()
    try:
        run(args.dataset, args.output, args.model, include_images=not args.no_images)
    except (FileNotFoundError, ValueError, RuntimeError, OSError, OpenAIError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())