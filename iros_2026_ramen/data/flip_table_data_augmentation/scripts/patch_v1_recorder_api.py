#!/usr/bin/env python3
"""Restore the demo-id argument dropped by the RoboFinals V1 recorder wrapper."""

from __future__ import annotations

import os
from pathlib import Path


MARKER = "# FLIP_TABLE_RECORDER_DEMO_IDS_COMPAT_V1"
SIGNATURE_BEFORE = "    def export_episodes(self, env_ids=None) -> None:\n"
SIGNATURE_AFTER = (
    "    def export_episodes(self, env_ids=None, demo_ids=None) -> None:\n"
    f"        {MARKER}\n"
)
CALL_BEFORE = "        orig_export_episodes(self, env_ids)\n"
CALL_AFTER = "        orig_export_episodes(self, env_ids, demo_ids=demo_ids)\n"


def patch_recorder_api(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if MARKER in source:
        if SIGNATURE_AFTER not in source or CALL_AFTER not in source:
            raise RuntimeError(f"incomplete recorder compatibility patch: {path}")
        return False
    if source.count(SIGNATURE_BEFORE) != 1 or source.count(CALL_BEFORE) != 1:
        raise RuntimeError(f"unsupported RoboFinals V1 recorder wrapper: {path}")
    patched = source.replace(SIGNATURE_BEFORE, SIGNATURE_AFTER).replace(
        CALL_BEFORE,
        CALL_AFTER,
    )
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    root = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))
    path = root / "robofinals/utils/monkey_patch.py"
    changed = patch_recorder_api(path)
    state = "patched" if changed else "verified"
    print(f"[flip_table] {state} V1 recorder demo-id forwarding: {path}")


if __name__ == "__main__":
    main()
