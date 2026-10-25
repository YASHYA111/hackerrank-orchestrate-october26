from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class IdentitySet:
    user_ids: set[str] = field(default_factory=set)
    emails: set[str] = field(default_factory=set)
    phones: set[str] = field(default_factory=set)
    email_hashes: set[str] = field(default_factory=set)

    def all_values(self) -> list[str]:
        return sorted(self.user_ids | self.emails | self.phones | self.email_hashes, key=str.casefold)


class TemporalIdentityResolver:
    def __init__(
        self,
        identities_path: Path,
        audit_path: Path,
        activity_path: Path | None = None,
    ) -> None:
        self.identities = pd.read_csv(identities_path, dtype=str, keep_default_na=False)
        self.audit_log = pd.read_csv(audit_path, dtype=str, keep_default_na=False)
        self.activities = (
            pd.read_csv(activity_path, dtype=str, keep_default_na=False)
            if activity_path is not None
            else pd.DataFrame()
        )
        self._history: dict[str, list[tuple[datetime | None, str, str]]] = {}
        self._links: list[tuple[datetime | None, str, str]] = []
        self._index_identity_history()

    def _index_identity_history(self) -> None:
        for row in self.identities.to_dict(orient="records"):
            user_id = row.get("user_id", "").strip()
            if not user_id:
                continue
            updated_at = self._time(row.get("updated_at", ""))
            for field_name in ("email", "phone", "email_hash"):
                value = row.get(field_name, "").strip()
                if value:
                    self._history.setdefault(user_id, []).append((updated_at, field_name, value))

        for row in self.audit_log.to_dict(orient="records"):
            user_id = row.get("user_id", "").strip()
            if not user_id:
                continue
            field_name = row.get("field_name", "").strip().casefold()
            event_time = self._time(row.get("event_time", ""))
            old_value = row.get("old_value", "").strip()
            new_value = row.get("new_value", "").strip()
            if field_name in {"email", "email_address"}:
                identity_field = "email"
            elif field_name in {"phone", "phone_number"}:
                identity_field = "phone"
            elif field_name in {"email_hash", "email_sha256"}:
                identity_field = "email_hash"
            elif field_name in {"user_id", "merged_user_id", "previous_user_id"}:
                if old_value and new_value:
                    self._links.append((event_time, old_value, new_value))
                continue
            else:
                continue
            history = self._history.setdefault(user_id, [])
            if old_value:
                history.append((event_time, identity_field, old_value))
            if new_value:
                history.append((event_time, identity_field, new_value))

        for row in self.activities.to_dict(orient="records"):
            if row.get("event_type", "").strip().upper() != "LINK_RESOURCE_IDENTITY":
                continue
            details = row.get("details", "")
            event_time = self._time(row.get("event_time", ""))
            user_ids = set(re.findall(r"\buser_[A-Za-z0-9_-]+\b", details, re.IGNORECASE))
            emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", details))
            for user_id in user_ids:
                for email in emails:
                    self._links.append((event_time, user_id, email))

    @staticmethod
    def _time(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed

    def resolve(self, email: str, audit_at: str | None = None) -> IdentitySet:
        cutoff = self._time(audit_at) if audit_at is not None else None
        if audit_at is not None and cutoff is None:
            raise ValueError(f"Invalid audit_at timestamp: {audit_at!r}")
        graph: dict[str, set[str]] = {}
        for event_time, left, right in self._links:
            if cutoff is None or (event_time is not None and event_time <= cutoff):
                left_key = self._key(left)
                right_key = self._key(right)
                graph.setdefault(left_key, set()).add(right_key)
                graph.setdefault(right_key, set()).add(left_key)
        for user_id, history in self._history.items():
            for event_time, _, value in history:
                if cutoff is None or (event_time is not None and event_time <= cutoff):
                    alias_key = self._key(value)
                    graph.setdefault(user_id, set()).add(alias_key)
                    graph.setdefault(alias_key, set()).add(user_id)
        pending = [self._key(email.strip())]
        visited: set[str] = set()
        while pending:
            value = pending.pop()
            if value in visited:
                continue
            visited.add(value)
            pending.extend(graph.get(value, set()) - visited)

        result = IdentitySet()
        for value in visited:
            if value.startswith("user_"):
                result.user_ids.add(value)
            elif "@" in value:
                result.emails.add(value)
            elif value and re.fullmatch(r"[0-9a-fA-F]{64}", value):
                result.email_hashes.add(value)
            elif value:
                result.phones.add(value)
        if not result.emails and email.strip():
            result.emails.add(email.strip())
        return result

    @staticmethod
    def _key(value: str) -> str:
        return value.casefold() if "@" in value else value

    def get_all_identities(self, email: str) -> list[str]:
        return self.resolve(email).all_values()
