from __future__ import annotations

import base64
import json
import re
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from identity_resolver import IdentitySet


@dataclass
class EvidenceBundle:
    text_evidence: list[dict[str, str]] = field(default_factory=list)
    images: list[dict[str, str]] = field(default_factory=list)
    activities: list[dict[str, str]] = field(default_factory=list)


class EvidenceAggregator:
    def __init__(
        self,
        dataset_dir: Path,
        manifest_path: Path,
        activity_path: Path,
        include_images: bool = True,
    ) -> None:
        self.dataset_dir = dataset_dir.resolve()
        self.manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        self.activities = pd.read_csv(activity_path, dtype=str, keep_default_na=False)
        self.include_images = include_images
        self._artifacts: list[tuple[str, str, str, str]] | None = None

    @staticmethod
    def _time(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed

    @staticmethod
    def _text_matches(content: str, identifiers: set[str]) -> bool:
        folded = content.casefold()
        for identifier in identifiers:
            needle = identifier.strip().casefold()
            if not needle:
                continue
            digits = re.sub(r"\D", "", needle)
            if len(digits) >= 7 and "@" not in needle and not re.search(r"[a-z_]", needle):
                pattern = r"(?<!\d)" + r"\D*".join(re.escape(digit) for digit in digits) + r"(?!\d)"
                if re.search(pattern, folded):
                    return True
                continue
            if re.fullmatch(r"[\w.@+-]+", needle):
                if re.search(rf"(?<![\w.@+-]){re.escape(needle)}(?![\w.@+-])", folded):
                    return True
            elif needle in folded:
                return True
        return False

    @staticmethod
    def _nearby_lines(content: str, identifiers: set[str], limit: int = 40) -> str:
        lines = content.splitlines()
        matched: list[str] = []
        for index, line in enumerate(lines):
            if EvidenceAggregator._text_matches(line, identifiers):
                matched.extend(lines[max(0, index - 1) : min(len(lines), index + 2)])
        unique = list(dict.fromkeys(line.strip() for line in matched if line.strip()))
        return "\n".join(unique[:limit])

    @staticmethod
    def _matching_lines(content: str, identifiers: set[str]) -> str:
        return "\n".join(
            line.strip()
            for line in content.splitlines()
            if line.strip() and EvidenceAggregator._text_matches(line, identifiers)
        )

    @staticmethod
    def _json_matches(content: str, identifiers: set[str]) -> str:
        try:
            document = json.loads(content)
        except json.JSONDecodeError:
            return EvidenceAggregator._nearby_lines(content, identifiers)

        context_keys = {
            "data_store",
            "collection",
            "captured_at",
            "origin_system",
            "system_id",
            "table",
            "resource_id",
            "customer_reference",
        }
        matches: list[str] = []

        def visit(value: Any, context: dict[str, Any]) -> None:
            if isinstance(value, dict):
                current_context = {
                    **context,
                    **{key: item for key, item in value.items() if key in context_keys and not isinstance(item, (dict, list))},
                }
                scalar_fields = {
                    key: item for key, item in value.items() if not isinstance(item, (dict, list))
                }
                if scalar_fields and EvidenceAggregator._text_matches(
                    json.dumps(scalar_fields, ensure_ascii=False, default=str), identifiers
                ):
                    matches.append(
                        json.dumps(
                            {"context": current_context, "record": scalar_fields},
                            ensure_ascii=False,
                            default=str,
                        )
                    )
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        visit(child, current_context)
            elif isinstance(value, list):
                for child in value:
                    visit(child, context)

        visit(document, {})
        return "\n".join(dict.fromkeys(matches[:100]))

    def _safe_path(self, relative: str) -> Path:
        path = (self.dataset_dir / relative).resolve()
        if self.dataset_dir not in path.parents:
            raise ValueError(f"Evidence path escapes dataset directory: {relative}")
        return path

    def _read_sql_snapshot(self, path: Path) -> str:
        database = sqlite3.connect(":memory:")
        try:
            database.executescript(path.read_text(encoding="utf-8-sig"))
            tables = database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            chunks: list[str] = []
            for (table_name,) in tables:
                quoted_table = table_name.replace('"', '""')
                columns = database.execute(f'PRAGMA table_info("{quoted_table}")').fetchall()
                column_names = [column[1] for column in columns]
                rows = database.execute(f'SELECT * FROM "{quoted_table}"').fetchall()
                chunks.extend(
                    json.dumps(
                        {"table": table_name, "columns": column_names, "row": row},
                        ensure_ascii=False,
                        default=str,
                    )
                    for row in rows
                )
            return "\n".join(chunks)
        finally:
            database.close()

    def _load_artifacts(self) -> list[tuple[str, str, str, str]]:
        if self._artifacts is not None:
            return self._artifacts
        artifacts: list[tuple[str, str, str, str]] = []
        for row in self.manifest.to_dict(orient="records"):
            file_id = row["file_id"]
            relative_path = row["file_path"]
            captured_at = row["captured_at"]
            path = self._safe_path(relative_path)
            if not path.is_file():
                warnings.warn(f"Evidence file {file_id} is missing: {path}", RuntimeWarning)
                continue

            suffix = path.suffix.casefold()
            if suffix in {".sql", ".sqlite", ".db"}:
                content = self._read_sql_snapshot(path) if suffix == ".sql" else self._read_database(path)
            elif suffix == ".pdf":
                content = self._read_pdf(path)
            elif suffix in {".json", ".log", ".txt", ".csv"}:
                content = path.read_text(encoding="utf-8-sig", errors="replace")
            elif suffix == ".png":
                content = ""
            else:
                warnings.warn(f"Unsupported evidence format for {file_id}: {suffix}", RuntimeWarning)
                continue
            artifacts.append((file_id, relative_path, captured_at, content))
        self._artifacts = artifacts
        return artifacts

    @staticmethod
    def _read_database(path: Path) -> str:
        uri = f"file:{path.as_posix()}?mode=ro"
        database = sqlite3.connect(uri, uri=True)
        try:
            tables = database.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            chunks = []
            for (table_name,) in tables:
                quoted = table_name.replace('"', '""')
                columns = database.execute(f'PRAGMA table_info("{quoted}")').fetchall()
                rows = database.execute(f'SELECT * FROM "{quoted}"').fetchall()
                chunks.extend(
                    json.dumps(
                        {"table": table_name, "columns": [column[1] for column in columns], "row": row},
                        ensure_ascii=False,
                        default=str,
                    )
                    for row in rows
                )
            return "\n".join(chunks)
        finally:
            database.close()

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            warnings.warn(f"Cannot extract PDF text without pypdf: {path.name}", RuntimeWarning)
            return ""
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def collect(self, identities: IdentitySet, audit_at: str) -> EvidenceBundle:
        cutoff = self._time(audit_at)
        if cutoff is None:
            raise ValueError(f"Invalid audit_at timestamp: {audit_at!r}")
        identifiers = {value for value in identities.all_values() if value}
        bundle = EvidenceBundle()
        resource_ids: set[str] = set()

        for file_id, relative_path, captured_at, content in self._load_artifacts():
            captured = self._time(captured_at)
            if cutoff is not None and (captured is None or captured > cutoff):
                continue
            path = self._safe_path(relative_path)
            if path.suffix.casefold() == ".png":
                if self.include_images:
                    bundle.images.append(
                        {
                            "evidence_id": file_id,
                            "file_name": path.name,
                            "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                        }
                    )
                continue
            if not content or not self._text_matches(content, identifiers):
                continue
            snippet = (
                self._json_matches(content, identifiers)
                if path.suffix.casefold() == ".json"
                else (
                    self._matching_lines(content, identifiers)
                    if path.suffix.casefold() in {".sql", ".sqlite", ".db"}
                    else self._nearby_lines(content, identifiers)
                )
            )
            if snippet:
                resource_ids.update(re.findall(r"\b(?:account|search|analytics|support|export|audit|tombstone|backup|data_flow|cache|order|message|attachment)_\d+\b", snippet, re.I))
                bundle.text_evidence.append(
                    {
                        "evidence_id": file_id,
                        "file_name": path.name,
                        "captured_at": captured_at,
                        "content": snippet,
                    }
                )

        for activity in self.activities.to_dict(orient="records"):
            event_time = self._time(activity.get("event_time", ""))
            if cutoff is not None and (event_time is None or event_time > cutoff):
                continue
            text = " ".join(activity.values())
            if self._text_matches(text, identifiers) or activity.get("resource_id") in resource_ids:
                bundle.activities.append(activity)

        return bundle
