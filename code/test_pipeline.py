from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from deterministic_router import DeterministicRouter
from evidence_aggregator import EvidenceAggregator
from identity_resolver import TemporalIdentityResolver
from perception_engine import ActiveFlow, EvidenceRecord, PerceptionResult, VerifiedState


ROOT = Path(__file__).resolve().parents[1]


class TemporalIdentityResolverTests(unittest.TestCase):
    def test_resolves_historical_aliases_and_obeys_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            identities = directory / "identities.csv"
            audit = directory / "audit.csv"
            activity = directory / "activity.csv"
            self._write_csv(
                identities,
                ["user_id", "email", "phone", "email_hash", "updated_at"],
                [["user_01", "new@example.test", "555-2222", "b" * 64, "2026-10-01T00:00:00Z"]],
            )
            self._write_csv(
                audit,
                ["audit_id", "user_id", "event_time", "field_name", "old_value", "new_value"],
                [
                    ["audit_1", "user_01", "2026-01-01T00:00:00Z", "email", "old@example.test", "new@example.test"],
                    ["audit_2", "user_01", "2026-12-01T00:00:00Z", "phone", "555-2222", "555-9999"],
                ],
            )
            self._write_csv(
                activity,
                ["activity_id", "event_time", "event_type", "details"],
                [["activity_1", "2026-09-01T00:00:00Z", "LINK_RESOURCE_IDENTITY", "Linked user_02 to old@example.test"]],
            )
            resolver = TemporalIdentityResolver(identities, audit, activity)

            resolved = resolver.resolve("OLD@example.test", "2026-10-02T00:00:00Z")

            self.assertEqual(resolved.user_ids, {"user_01", "user_02"})
            self.assertIn("old@example.test", {value.casefold() for value in resolved.emails})
            self.assertNotIn("555-9999", resolved.phones)

    @staticmethod
    def _write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(columns)
            writer.writerows(rows)


class EvidenceAggregatorTests(unittest.TestCase):
    def test_json_matches_include_only_the_matching_record(self) -> None:
        content = (
            '{"snapshots":[{"data_store":"account_service","records":['
            '{"resource_id":"account_1","email":"target@example.test"},'
            '{"resource_id":"account_2","email":"other@example.test"}]}]}'
        )

        result = EvidenceAggregator._json_matches(content, {"target@example.test"})

        self.assertIn("account_1", result)
        self.assertNotIn("account_2", result)


class DeterministicRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = DeterministicRouter(str(ROOT / "dataset" / "deletion_rules.csv"))

    def test_non_deletion_request_is_not_applicable(self) -> None:
        result = self.router.decide(PerceptionResult(intent_is_deletion=False))

        self.assertEqual(result.status, "NOT_APPLICABLE")
        self.assertEqual(result.steps, [])

    def test_complete_requires_evidence_backed_verification(self) -> None:
        result = self.router.decide(
            PerceptionResult(
                intent_is_deletion=True,
                verified_states=[
                    VerifiedState(
                        system_id="account_service",
                        resource_id="account_1",
                        evidence_id="evidence_1",
                        data_category="USER_PROFILE",
                        policy_condition_met=True,
                        state="ABSENT",
                    )
                ],
            )
        )

        self.assertEqual(result.status, "COMPLETE")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0]["operation"], "VERIFY_RECORD_ABSENT")
        self.assertEqual(result.steps[0]["evidence_id"], "evidence_1")

    def test_flows_are_remediated_before_tombstone_redaction(self) -> None:
        perception = PerceptionResult(
            intent_is_deletion=True,
            records_found=[
                EvidenceRecord(
                    system_id="identity_service",
                    resource_id="tombstone_1",
                    evidence_id="evidence_1",
                    data_category="ACCOUNT_TOMBSTONE",
                    is_tombstone=True,
                    pii_exposed=True,
                    policy_condition_met=True,
                    observed_state="PRESENT",
                )
            ],
            active_flows=[
                ActiveFlow(
                    system_id="integration_service",
                    resource_id="flow_1",
                    evidence_id="activity_1",
                    flow_type="SYNC",
                )
            ],
        )

        result = self.router.decide(perception)
        steps = result.steps

        self.assertEqual(result.status, "INCOMPLETE")
        self.assertEqual(
            [step["phase"] for step in steps],
            ["FINDING", "FINDING", "REMEDIATION", "REMEDIATION", "VERIFICATION", "VERIFICATION"],
        )
        remediation = [step for step in steps if step["phase"] == "REMEDIATION"]
        self.assertEqual(
            [step["operation"] for step in remediation],
            ["DISABLE_DATA_FLOW", "REDACT_RECORD"],
        )
        self.assertTrue(all(step["evidence_id"] == "" for step in remediation))
        self.assertTrue(
            all(step["evidence_id"] == "" for step in steps if step["phase"] == "VERIFICATION")
        )
        self.assertEqual([step["step_no"] for step in steps], list(range(1, len(steps) + 1)))


if __name__ == "__main__":
    unittest.main()
