"""Crash-safe trajectory and appearance-variant provenance records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

from .config import canonical_json_digest
from .io_utils import atomic_write_json


LEDGER_SCHEMA_VERSION = "team_ramen_flip_table_candidate/v2"
APPEARANCE_SCHEMA_VERSION = "team_ramen_flip_table_appearance/v1"
FINAL_STATES = frozenset({"rejected", "exported"})
ALLOWED_TRANSITIONS = {
    "claimed": frozenset({"generated", "rejected"}),
    "generated": frozenset({"validated", "rejected"}),
    "validated": frozenset({"rendered", "rejected"}),
    "rendered": frozenset({"exported", "rejected"}),
    "rejected": frozenset(),
    "exported": frozenset(),
}
APPEARANCE_TRANSITIONS = {
    "claimed": frozenset({"rendered", "rejected"}),
    "rendered": frozenset({"exported", "rejected"}),
    "rejected": frozenset(),
    "exported": frozenset(),
}
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
def _validate_sources(value: Any, *, allow_empty: bool) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or (not value and not allow_empty)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value)
        or len(value) != len(set(value))
    ):
        qualifier = "possibly empty" if allow_empty else "non-empty"
        raise ValueError(
            f"source_episode_indices must be a {qualifier} list of unique non-negative integers"
        )
    return tuple(value)


def _validate_common_hashes(value: dict[str, Any]) -> None:
    if not isinstance(value.get("config_sha256"), str) or not _SHA256.fullmatch(
        value["config_sha256"]
    ):
        raise ValueError("config_sha256 must be a lowercase SHA-256")
    if not isinstance(value.get("runtime_digest"), str) or not _SHA256.fullmatch(
        value["runtime_digest"]
    ):
        raise ValueError("runtime_digest must be a lowercase SHA-256")


@dataclass(frozen=True)
class CandidateRecord:
    """One physical trajectory attempt.

    Source episodes are not known when the attempt is claimed. They are bound
    exactly once by the ``claimed -> generated`` transition.
    """

    candidate_id: str
    status: str
    source_episode_indices: tuple[int, ...]
    trajectory_seed: int
    config_sha256: str
    runtime_digest: str
    payload: dict[str, Any]

    @classmethod
    def from_json(cls, value: Any) -> "CandidateRecord":
        if not isinstance(value, dict) or value.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise ValueError("unsupported candidate record schema")
        candidate_id = value.get("candidate_id")
        status = value.get("status")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError("invalid candidate_id")
        if status not in ALLOWED_TRANSITIONS:
            raise ValueError(f"invalid candidate status: {status!r}")
        sources = _validate_sources(
            value.get("source_episode_indices"), allow_empty=status == "claimed"
        )
        trajectory_seed = value.get("trajectory_seed")
        if isinstance(trajectory_seed, bool) or not isinstance(trajectory_seed, int) or trajectory_seed < 0:
            raise ValueError("trajectory_seed must be a non-negative integer")
        _validate_common_hashes(value)
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return cls(
            candidate_id=candidate_id,
            status=status,
            source_episode_indices=sources,
            trajectory_seed=trajectory_seed,
            config_sha256=value["config_sha256"],
            runtime_digest=value["runtime_digest"],
            payload=payload,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "source_episode_indices": list(self.source_episode_indices),
            "trajectory_seed": self.trajectory_seed,
            "config_sha256": self.config_sha256,
            "runtime_digest": self.runtime_digest,
            "payload": self.payload,
        }


@dataclass(frozen=True)
class AppearanceRecord:
    """One deterministic visual rendering of an accepted physical trajectory."""

    candidate_id: str
    variant_index: int
    status: str
    appearance_seed: int
    trajectory_sha256: str
    config_sha256: str
    runtime_digest: str
    payload: dict[str, Any]

    @classmethod
    def from_json(cls, value: Any) -> "AppearanceRecord":
        if not isinstance(value, dict) or value.get("schema_version") != APPEARANCE_SCHEMA_VERSION:
            raise ValueError("unsupported appearance record schema")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError("invalid candidate_id")
        variant_index = value.get("variant_index")
        appearance_seed = value.get("appearance_seed")
        for name, item in (("variant_index", variant_index), ("appearance_seed", appearance_seed)):
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        status = value.get("status")
        if status not in APPEARANCE_TRANSITIONS:
            raise ValueError(f"invalid appearance status: {status!r}")
        trajectory_sha256 = value.get("trajectory_sha256")
        if not isinstance(trajectory_sha256, str) or not _SHA256.fullmatch(trajectory_sha256):
            raise ValueError("trajectory_sha256 must be a lowercase SHA-256")
        _validate_common_hashes(value)
        payload = value.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return cls(
            candidate_id=candidate_id,
            variant_index=variant_index,
            status=status,
            appearance_seed=appearance_seed,
            trajectory_sha256=trajectory_sha256,
            config_sha256=value["config_sha256"],
            runtime_digest=value["runtime_digest"],
            payload=payload,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": APPEARANCE_SCHEMA_VERSION,
            "candidate_id": self.candidate_id,
            "variant_index": self.variant_index,
            "status": self.status,
            "appearance_seed": self.appearance_seed,
            "trajectory_sha256": self.trajectory_sha256,
            "config_sha256": self.config_sha256,
            "runtime_digest": self.runtime_digest,
            "payload": self.payload,
        }


class CandidateLedger:
    """Atomically persisted records for physical attempts and render variants."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.records = self.root / "records"
        self.variants = self.root / "appearance_variants"

    def path_for(self, candidate_id: str) -> Path:
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError("invalid candidate_id")
        return self.records / f"{candidate_id}.json"

    def variant_path_for(self, candidate_id: str, variant_index: int) -> Path:
        if not _CANDIDATE_ID.fullmatch(candidate_id):
            raise ValueError("invalid candidate_id")
        if isinstance(variant_index, bool) or not isinstance(variant_index, int) or variant_index < 0:
            raise ValueError("variant_index must be a non-negative integer")
        return self.variants / candidate_id / f"variant-{variant_index:04d}.json"

    @staticmethod
    def _create_exclusive(path: Path, payload: dict[str, Any], kind: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as exc:
            raise FileExistsError(f"{kind} already exists: {path.stem}") from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def claim(self, record: CandidateRecord) -> None:
        if record.status != "claimed" or record.source_episode_indices:
            raise ValueError("new candidates must be claimed before source selection")
        self._create_exclusive(self.path_for(record.candidate_id), record.to_json(), "candidate")

    def ensure_claim(self, record: CandidateRecord) -> CandidateRecord:
        """Claim a new attempt or verify an identical existing attempt identity."""

        try:
            self.claim(record)
            return record
        except FileExistsError:
            current = self.load(record.candidate_id)
            identity = (
                current.candidate_id,
                current.trajectory_seed,
                current.config_sha256,
                current.runtime_digest,
            )
            expected = (
                record.candidate_id,
                record.trajectory_seed,
                record.config_sha256,
                record.runtime_digest,
            )
            stable_claim_payload = {
                key: value for key, value in record.payload.items() if key != "started_at_utc"
            }
            if identity != expected or any(
                current.payload.get(key) != value for key, value in stable_claim_payload.items()
            ):
                raise ValueError(f"candidate identity mismatch on resume: {record.candidate_id}")
            return current

    def load(self, candidate_id: str) -> CandidateRecord:
        return CandidateRecord.from_json(
            json.loads(self.path_for(candidate_id).read_text(encoding="utf-8"))
        )

    def transition(
        self,
        candidate_id: str,
        status: str,
        payload: dict[str, Any],
        *,
        source_episode_indices: tuple[int, ...] | None = None,
    ) -> CandidateRecord:
        current = self.load(candidate_id)
        if status not in ALLOWED_TRANSITIONS[current.status]:
            raise ValueError(f"invalid transition {current.status!r} -> {status!r}")
        if source_episode_indices is not None:
            if current.status != "claimed" or status != "generated" or current.source_episode_indices:
                raise ValueError("source episodes may only be bound while marking a claim generated")
            sources = _validate_sources(list(source_episode_indices), allow_empty=False)
        else:
            sources = current.source_episode_indices
        if status != "claimed" and not sources:
            raise ValueError("generated candidates must bind at least one source episode")
        merged = dict(current.payload)
        overlap = set(merged).intersection(payload)
        if overlap:
            raise ValueError(f"candidate payload keys are immutable: {sorted(overlap)}")
        merged.update(payload)
        updated = CandidateRecord(
            candidate_id=current.candidate_id,
            status=status,
            source_episode_indices=sources,
            trajectory_seed=current.trajectory_seed,
            config_sha256=current.config_sha256,
            runtime_digest=current.runtime_digest,
            payload=merged,
        )
        atomic_write_json(self.path_for(candidate_id), updated.to_json())
        return updated

    def claim_variant(self, record: AppearanceRecord) -> None:
        if record.status != "claimed":
            raise ValueError("new appearance records must start as claimed")
        candidate = self.load(record.candidate_id)
        if candidate.status not in {"validated", "rendered", "exported"}:
            raise ValueError("appearance variants require a validated physical trajectory")
        expected_hash = candidate.payload.get("trajectory_sha256")
        if expected_hash != record.trajectory_sha256:
            raise ValueError("appearance trajectory hash does not match its candidate")
        if (
            candidate.config_sha256 != record.config_sha256
            or candidate.runtime_digest != record.runtime_digest
        ):
            raise ValueError("appearance runtime identity does not match its candidate")
        self._create_exclusive(
            self.variant_path_for(record.candidate_id, record.variant_index),
            record.to_json(),
            "appearance variant",
        )

    def ensure_variant_claim(self, record: AppearanceRecord) -> AppearanceRecord:
        try:
            self.claim_variant(record)
            return record
        except FileExistsError:
            current = self.load_variant(record.candidate_id, record.variant_index)
            expected = (
                record.candidate_id,
                record.variant_index,
                record.appearance_seed,
                record.trajectory_sha256,
                record.config_sha256,
                record.runtime_digest,
            )
            actual = (
                current.candidate_id,
                current.variant_index,
                current.appearance_seed,
                current.trajectory_sha256,
                current.config_sha256,
                current.runtime_digest,
            )
            if actual != expected:
                raise ValueError("appearance identity mismatch on resume")
            return current

    def load_variant(self, candidate_id: str, variant_index: int) -> AppearanceRecord:
        return AppearanceRecord.from_json(
            json.loads(self.variant_path_for(candidate_id, variant_index).read_text(encoding="utf-8"))
        )

    def transition_variant(
        self,
        candidate_id: str,
        variant_index: int,
        status: str,
        payload: dict[str, Any],
    ) -> AppearanceRecord:
        current = self.load_variant(candidate_id, variant_index)
        if status not in APPEARANCE_TRANSITIONS[current.status]:
            raise ValueError(f"invalid appearance transition {current.status!r} -> {status!r}")
        merged = dict(current.payload)
        overlap = set(merged).intersection(payload)
        if overlap:
            raise ValueError(f"appearance payload keys are immutable: {sorted(overlap)}")
        merged.update(payload)
        updated = AppearanceRecord(
            candidate_id=current.candidate_id,
            variant_index=current.variant_index,
            status=status,
            appearance_seed=current.appearance_seed,
            trajectory_sha256=current.trajectory_sha256,
            config_sha256=current.config_sha256,
            runtime_digest=current.runtime_digest,
            payload=merged,
        )
        atomic_write_json(self.variant_path_for(candidate_id, variant_index), updated.to_json())
        return updated

    def complete_rendering(self, candidate_id: str, minimum_variants: int) -> CandidateRecord:
        if minimum_variants <= 0:
            raise ValueError("minimum_variants must be positive")
        candidate = self.load(candidate_id)
        if candidate.status != "validated":
            raise ValueError("only a validated candidate can finish rendering")
        variants = self.list_variants(candidate_id)
        rendered = [item for item in variants if item.status in {"rendered", "exported"}]
        if len(rendered) < minimum_variants:
            raise ValueError(
                f"candidate {candidate_id} has {len(rendered)} rendered variants; "
                f"requires {minimum_variants}"
            )
        indices = [item.variant_index for item in rendered]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate appearance variant indices")
        return self.transition(
            candidate_id,
            "rendered",
            {
                "rendered_variant_indices": sorted(indices),
                "appearance_manifest_sha256": canonical_json_digest(
                    [item.to_json() for item in sorted(rendered, key=lambda item: item.variant_index)]
                ),
            },
        )

    def list_records(self) -> tuple[CandidateRecord, ...]:
        if not self.records.is_dir():
            return ()
        return tuple(self.load(path.stem) for path in sorted(self.records.glob("*.json")))

    def list_variants(self, candidate_id: str | None = None) -> tuple[AppearanceRecord, ...]:
        if not self.variants.is_dir():
            return ()
        paths = (
            sorted((self.variants / candidate_id).glob("variant-*.json"))
            if candidate_id is not None
            else sorted(self.variants.glob("*/variant-*.json"))
        )
        values = []
        for path in paths:
            variant_candidate_id = path.parent.name
            variant_index = int(path.stem.removeprefix("variant-"))
            values.append(self.load_variant(variant_candidate_id, variant_index))
        return tuple(values)

    def manifest_digest(self) -> str:
        return canonical_json_digest(
            {
                "candidates": [record.to_json() for record in self.list_records()],
                "appearance_variants": [record.to_json() for record in self.list_variants()],
            }
        )
