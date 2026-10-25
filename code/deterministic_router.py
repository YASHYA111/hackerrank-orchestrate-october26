from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from perception_engine import PerceptionResult


PHASE_ORDER = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
OPERATION_ORDER = {
    "REPORT_REMAINING_RECORD": 0,
    "REPORT_EXPOSED_PII": 1,
    "REPORT_ACTIVE_DATA_FLOW": 2,
    "REPORT_RESTORE_RISK": 3,
    "DISABLE_DATA_FLOW": 4,
    "BLOCK_RESTORE": 5,
    "DELETE_RECORD": 6,
    "REDACT_RECORD": 7,
    "ANONYMIZE_RECORD": 8,
    "VERIFY_RECORD_ABSENT": 9,
    "VERIFY_RECORD_PRESENT": 10,
    "VERIFY_RECORD_REDACTED": 11,
    "VERIFY_RECORD_ANONYMIZED": 12,
    "VERIFY_DATA_FLOW_DISABLED": 13,
    "VERIFY_RESTORE_BLOCKED": 14,
}
REMEDIATION = {
    "DELETE": ("DELETE_RECORD", "ABSENT", "VERIFY_RECORD_ABSENT", "ABSENT"),
    "REDACT": ("REDACT_RECORD", "REDACTED", "VERIFY_RECORD_REDACTED", "REDACTED"),
    "ANONYMIZE": ("ANONYMIZE_RECORD", "ANONYMIZED", "VERIFY_RECORD_ANONYMIZED", "ANONYMIZED"),
    "RETAIN": (None, None, "VERIFY_RECORD_PRESENT", "PRESENT"),
}


@dataclass
class CaseDecision:
    status: str
    steps: list[dict[str, Any]]


class DeterministicRouter:
    def __init__(self, rules_path: str) -> None:
        self.rules = pd.read_csv(rules_path, dtype=str, keep_default_na=False)
        self._actions = {
            row["data_category"]: row["required_action"].strip().upper()
            for row in self.rules.to_dict(orient="records")
        }
        self._conditions = {
            row["data_category"]: row["condition"].strip()
            for row in self.rules.to_dict(orient="records")
        }

    def _condition_met(self, category: str, evidence_says_met: bool) -> bool:
        return self._conditions.get(category, "") == "Always" or evidence_says_met

    @staticmethod
    def _step(
        phase: str,
        system_id: str,
        resource_id: str,
        operation: str,
        state: str,
        evidence_id: str,
    ) -> dict[str, Any]:
        return {
            "step_no": 0,
            "phase": phase,
            "system_id": system_id,
            "resource_id": resource_id,
            "operation": operation,
            "state": state,
            "evidence_id": evidence_id,
        }

    def decide(self, perception: PerceptionResult) -> CaseDecision:
        if not perception.intent_is_deletion:
            return CaseDecision("NOT_APPLICABLE", [])

        findings: list[dict[str, Any]] = []
        flow_steps: list[dict[str, Any]] = []
        record_steps: list[dict[str, Any]] = []
        verification_steps: list[dict[str, Any]] = []

        for flow in perception.active_flows:
            if flow.flow_type == "SYNC":
                findings.append(
                    self._step("FINDING", flow.system_id, flow.resource_id, "REPORT_ACTIVE_DATA_FLOW", "ACTIVE", flow.evidence_id)
                )
                flow_steps.append(
                    self._step("REMEDIATION", flow.system_id, flow.resource_id, "DISABLE_DATA_FLOW", "DISABLED", "")
                )
                verification_steps.append(
                    self._step("VERIFICATION", flow.system_id, flow.resource_id, "VERIFY_DATA_FLOW_DISABLED", "DISABLED", "")
                )
            else:
                findings.append(
                    self._step("FINDING", flow.system_id, flow.resource_id, "REPORT_RESTORE_RISK", "ACTIVE", flow.evidence_id)
                )
                flow_steps.append(
                    self._step("REMEDIATION", flow.system_id, flow.resource_id, "BLOCK_RESTORE", "BLOCKED", "")
                )
                verification_steps.append(
                    self._step("VERIFICATION", flow.system_id, flow.resource_id, "VERIFY_RESTORE_BLOCKED", "BLOCKED", "")
                )

        for record in perception.records_found:
            if record.observed_state == "UNKNOWN":
                raise ValueError(
                    f"Evidence does not establish the state of {record.system_id}/{record.resource_id}."
                )
            required_action = "REDACT" if record.is_tombstone else self._actions.get(record.data_category)
            if (
                not self._condition_met(record.data_category, record.policy_condition_met)
                and required_action in {"REDACT", "ANONYMIZE", "RETAIN"}
                and not record.is_tombstone
            ):
                required_action = "DELETE"
            if required_action is None:
                raise ValueError(f"No deletion rule found for category {record.data_category!r}.")
            if required_action not in REMEDIATION:
                raise ValueError(f"Unsupported deletion action {required_action!r} for {record.data_category!r}.")

            operation, remediation_state, verify_operation, verify_state = REMEDIATION[required_action]
            if required_action == "DELETE":
                violation = record.observed_state == "PRESENT"
                finding_operation = "REPORT_REMAINING_RECORD"
            elif required_action in {"REDACT", "ANONYMIZE"}:
                violation = record.observed_state == "PRESENT" and record.pii_exposed
                finding_operation = "REPORT_EXPOSED_PII"
            else:
                violation = record.observed_state != "PRESENT"
                finding_operation = "REPORT_REMAINING_RECORD"

            if violation:
                findings.append(
                    self._step("FINDING", record.system_id, record.resource_id, finding_operation, "PRESENT", record.evidence_id)
                )
                if operation is not None and remediation_state is not None:
                    record_steps.append(
                        self._step("REMEDIATION", record.system_id, record.resource_id, operation, remediation_state, "")
                    )
                verification_steps.append(
                    self._step("VERIFICATION", record.system_id, record.resource_id, verify_operation, verify_state, "")
                )
            elif record.observed_state in {"ABSENT", "PRESENT", "REDACTED", "ANONYMIZED"}:
                verification_steps.append(
                    self._step("VERIFICATION", record.system_id, record.resource_id, verify_operation, verify_state, record.evidence_id)
                )

        for state in perception.verified_states:
            expected_action = self._actions.get(state.data_category)
            if state.data_category == "ACCOUNT_TOMBSTONE":
                expected_action = "REDACT"
            elif (
                not self._condition_met(state.data_category, state.policy_condition_met)
                and expected_action in {"REDACT", "ANONYMIZE", "RETAIN"}
            ):
                expected_action = "DELETE"
            if expected_action is None:
                raise ValueError(f"No deletion rule found for verified category {state.data_category!r}.")
            expected_operation = REMEDIATION[expected_action][2]
            expected_state = REMEDIATION[expected_action][3]
            if state.state != expected_state:
                raise ValueError(
                    f"Non-compliant state for {state.system_id}/{state.resource_id} must be reported in records_found."
                )
            verification_steps.append(
                self._step(
                    "VERIFICATION",
                    state.system_id,
                    state.resource_id,
                    expected_operation,
                    expected_state,
                    state.evidence_id,
                )
            )

        if findings:
            status = "INCOMPLETE"
            tasks = flow_steps + record_steps
            ordered_tasks = self._order_remediation(tasks)
            for step in verification_steps:
                step["evidence_id"] = ""
            combined = findings + ordered_tasks + verification_steps
        else:
            if not verification_steps:
                raise ValueError(
                    "No finding or evidence-backed verification was extracted; refusing to claim COMPLETE."
                )
            status = "COMPLETE"
            combined = verification_steps

        unique_steps: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        for step in combined:
            key = (
                step["phase"],
                step["system_id"],
                step["resource_id"],
                step["operation"],
                step["state"],
            )
            current = unique_steps.get(key)
            if current is None or step["evidence_id"] < current["evidence_id"]:
                unique_steps[key] = step
        ordered = sorted(
            unique_steps.values(),
            key=lambda step: (
                PHASE_ORDER[step["phase"]],
                OPERATION_ORDER[step["operation"]],
                step["system_id"],
                step["resource_id"],
                step["evidence_id"],
            ),
        )
        for step_no, step in enumerate(ordered, start=1):
            step["step_no"] = step_no
        return CaseDecision(status, ordered)

    @staticmethod
    def _order_remediation(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        graph = nx.DiGraph()
        for index, task in enumerate(tasks):
            graph.add_node(index, task=task)
        flow_nodes = [
            index
            for index, task in enumerate(tasks)
            if task["operation"] in {"DISABLE_DATA_FLOW", "BLOCK_RESTORE"}
        ]
        record_nodes = [
            index
            for index, task in enumerate(tasks)
            if task["operation"] in {"DELETE_RECORD", "REDACT_RECORD", "ANONYMIZE_RECORD"}
        ]
        graph.add_edges_from((flow, record) for flow in flow_nodes for record in record_nodes)
        ordered_nodes = nx.lexicographical_topological_sort(
            graph,
            key=lambda index: (
                OPERATION_ORDER[graph.nodes[index]["task"]["operation"]],
                graph.nodes[index]["task"]["system_id"],
                graph.nodes[index]["task"]["resource_id"],
            ),
        )
        return [graph.nodes[index]["task"] for index in ordered_nodes]
