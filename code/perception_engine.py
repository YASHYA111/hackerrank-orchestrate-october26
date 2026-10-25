from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from evidence_aggregator import EvidenceAggregator, EvidenceBundle
from identity_resolver import IdentitySet


class EvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    resource_id: str
    evidence_id: str
    data_category: str
    is_tombstone: bool
    pii_exposed: bool
    policy_condition_met: bool
    observed_state: Literal["PRESENT", "ABSENT", "REDACTED", "ANONYMIZED", "UNKNOWN"]


class ActiveFlow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    resource_id: str
    evidence_id: str
    flow_type: Literal["SYNC", "BACKUP"]


class VerifiedState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_id: str
    resource_id: str
    evidence_id: str
    data_category: str
    policy_condition_met: bool
    state: Literal["ABSENT", "PRESENT", "REDACTED", "ANONYMIZED"]


class PerceptionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent_is_deletion: bool
    records_found: list[EvidenceRecord] = Field(default_factory=list)
    active_flows: list[ActiveFlow] = Field(default_factory=list)
    verified_states: list[VerifiedState] = Field(default_factory=list)


class ImageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visible_text: str


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class PerceptionEngine:
    def __init__(self, model: str = "gpt-4o") -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run the GPT-4o perception engine.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.usage = Usage()
        self._image_summaries: dict[str, str] = {}

    def _summarize_image(self, image: dict[str, str]) -> str:
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Transcribe visible text faithfully. Preserve identifiers, resource IDs, system names, "
                        "dates, and status labels. Do not infer deletion status or add facts that are not visible."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Transcribe image evidence {image['evidence_id']}."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image['base64']}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            response_format=ImageSummary,
        )
        message = response.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"GPT-4o returned no image transcription: {message.refusal or 'empty parse'}")
        self.usage.calls += 1
        if response.usage:
            self.usage.input_tokens += response.usage.prompt_tokens
            self.usage.output_tokens += response.usage.completion_tokens
        return message.parsed.visible_text

    def analyze(
        self,
        request_text: str,
        identities: IdentitySet,
        evidence: EvidenceBundle,
        audit_at: str,
        policy_rules: list[dict[str, str]],
    ) -> PerceptionResult:
        identifiers = set(identities.all_values())
        image_evidence: list[dict[str, str]] = []
        included_image_ids: set[str] = set()
        for image in evidence.images:
            summary = self._image_summaries.get(image["evidence_id"])
            if summary is None:
                summary = self._summarize_image(image)
                self._image_summaries[image["evidence_id"]] = summary
            matching_text = EvidenceAggregator._nearby_lines(summary, identifiers)
            if matching_text:
                included_image_ids.add(image["evidence_id"])
                image_evidence.append(
                    {
                        "evidence_id": image["evidence_id"],
                        "file_name": image["file_name"],
                        "content": matching_text,
                    }
                )
        evidence.images = [
            image for image in evidence.images if image["evidence_id"] in included_image_ids
        ]
        text_payload = {
            "request_text": request_text,
            "audit_at": audit_at,
            "policy_rules": policy_rules,
            "identities": {
                "user_ids": sorted(identities.user_ids),
                "emails": sorted(identities.emails),
                "phones": sorted(identities.phones),
                "email_hashes": sorted(identities.email_hashes),
            },
            "text_evidence": evidence.text_evidence,
            "activity_events": evidence.activities,
            "image_evidence": image_evidence,
        }
        content: list[dict[str, object]] = [
            {
                "type": "text",
                "text": (
                    "Audit the supplied deletion request and evidence at audit_at. Determine intent from request_text. "
                    "Report only records and flows directly supported by the evidence. Treat activity and workflow "
                    "results as claims, not proof. Respect the audit cutoff: exclude later evidence and later state. "
                    "For each record, identify its system, resource, policy category, tombstone status, whether "
                    "prohibited identifiers are exposed, whether the condition in the matching deletion rule is "
                    "supported by evidence, and observed state. Use policy_rules to identify category conditions; "
                    "the deterministic router applies the action. Cite the exact evidence_id. "
                    "List verified_states only for compliant records whose required policy treatment is established "
                    "at the cutoff, including a complete scoped snapshot or direct verification. Put non-compliant "
                    "records in records_found. Never infer absence from no search hit. "
                    "Only label a flow active when evidence shows it is active at the cutoff. Do not invent IDs. "
                    "Evidence context follows:\n"
                    + json.dumps(text_payload, ensure_ascii=False)
                ),
            }
        ]
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an evidence extraction component, not a compliance decision-maker. "
                        "Do not rely on reported status. Every extracted record, flow, and verification must "
                        "cite supplied participant-facing evidence. Images are listed with evidence IDs; cite "
                        "those IDs for facts extracted from the corresponding image."
                    ),
                },
                {"role": "user", "content": content},
            ],
            response_format=PerceptionResult,
        )
        message = response.choices[0].message
        if message.parsed is None:
            raise RuntimeError(f"GPT-4o returned no structured perception result: {message.refusal or 'empty parse'}")
        self.usage.calls += 1
        if response.usage:
            self.usage.input_tokens += response.usage.prompt_tokens
            self.usage.output_tokens += response.usage.completion_tokens
        return message.parsed
