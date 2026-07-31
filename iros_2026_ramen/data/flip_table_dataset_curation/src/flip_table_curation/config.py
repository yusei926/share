from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .util import sha256_file, sha256_json


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "configs" / "curation_v2.yaml"


@dataclass(frozen=True)
class CurationConfig:
    path: Path
    raw: dict[str, Any]
    digest: str
    code_digest: str
    repo_root: Path
    package_root: Path
    workspace: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"missing config section {name!r}")
        return value

    def asset_path(self, key: str) -> Path:
        return (self.repo_root / str(self.section("assets")[key])).resolve()

    @property
    def source_repo_id(self) -> str:
        return str(self.section("source")["repo_id"])

    @property
    def source_revision(self) -> str:
        return str(self.section("source")["revision"])

    @property
    def target_repo_id(self) -> str:
        return str(self.section("target")["repo_id"])


def load_config(path: Path | None = None) -> CurationConfig:
    source = (path or DEFAULT_CONFIG_PATH).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("curation config must be a mapping")
    if raw.get("schema_version") != "team_ramen_flip_table_curation_config/v1":
        raise ValueError(f"unsupported schema_version: {raw.get('schema_version')!r}")
    workspace = PACKAGE_ROOT / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    code_files = {
        candidate.relative_to(PACKAGE_ROOT).as_posix(): sha256_file(candidate)
        for candidate in sorted((PACKAGE_ROOT / "src").rglob("*.py"))
    }
    return CurationConfig(
        path=source,
        raw=raw,
        digest=sha256_json(raw),
        code_digest=sha256_json(code_files),
        repo_root=REPO_ROOT,
        package_root=PACKAGE_ROOT,
        workspace=workspace,
    )
