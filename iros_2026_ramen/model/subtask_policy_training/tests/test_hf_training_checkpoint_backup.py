from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from model.subtask_policy_training.scripts import verify_hf_training_checkpoint as verifier


def _write_checkpoint(root: Path, step: int = 10_000) -> Path:
    checkpoint = root / f"{step:06d}"
    for relative in verifier.REQUIRED_FILES:
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fixture:{relative}".encode())
    return checkpoint


def _fake_api(checkpoint: Path, *, private: bool = True, omit: str | None = None):
    prefix = f"checkpoints/{checkpoint.name}/"
    siblings = []
    for relative in verifier.REQUIRED_FILES:
        if relative == omit:
            continue
        digest = hashlib.sha256((checkpoint / relative).read_bytes()).hexdigest()
        siblings.append(
            SimpleNamespace(
                rfilename=prefix + relative,
                lfs=SimpleNamespace(sha256=digest),
                blob_id=None,
            )
        )
    info = SimpleNamespace(private=private, siblings=siblings, sha="head-commit")
    refs = SimpleNamespace(
        tags=[
            SimpleNamespace(
                name=checkpoint.name,
                target_commit="tag-commit",
            )
        ]
    )
    return SimpleNamespace(
        model_info=lambda *_args, **_kwargs: info,
        list_repo_refs=lambda *_args, **_kwargs: refs,
    )


def test_resumable_checkpoint_backup_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    monkeypatch.setattr(verifier, "HfApi", lambda: _fake_api(checkpoint))
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_hf_training_checkpoint.py",
            "--repo-id",
            "Team-RAMEN/private-checkpoint",
            "--step",
            "10000",
            "--local-checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )

    verifier.main()

    receipt = json.loads(output.read_text())
    assert receipt["private"] is True
    assert receipt["resumable"] is True
    assert receipt["checkpoint_step"] == 10_000
    assert receipt["checkpoint_tag"] == "010000"
    assert set(receipt["hashes"]) == set(verifier.REQUIRED_FILES)


def test_checkpoint_backup_rejects_missing_optimizer_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    monkeypatch.setattr(
        verifier,
        "HfApi",
        lambda: _fake_api(
            checkpoint,
            omit="training_state/optimizer_state.safetensors",
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "verify_hf_training_checkpoint.py",
            "--repo-id",
            "Team-RAMEN/private-checkpoint",
            "--step",
            "10000",
            "--local-checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "receipt.json"),
        ],
    )

    with pytest.raises(FileNotFoundError, match="optimizer_state"):
        verifier.main()


def test_h100_runner_has_independent_resumable_backups() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts" / "run_h100_flip_table_groot_n17.sh").read_text()
    launcher = (root / "scripts" / "launch_h100_flip_table_groot_n17.sh").read_text()
    trainer = (root / "scripts" / "train_lerobot.sh").read_text()

    assert "GROOT_FULL_SAVE_FREQ:-5000" in runner
    assert "GROOT_TRAINING_TARGET:-both" in runner
    assert 'training_target" == "baseline"' in runner
    assert "prepare_training_view" in runner
    assert "team_ramen_training_view.json" in runner
    assert "--test-episodes" not in runner
    assert "_baseline_checkpoints" in runner
    assert "_auxiliary_checkpoints" in runner
    assert "verify_hf_training_checkpoint.py" in runner
    assert "SAVE_CHECKPOINT_TO_HUB=\"$durable_backup\"" in runner
    assert "GROOT_LOCAL_CHECKPOINT_KEEP:-1" in runner
    assert "HF_XET_HIGH_PERFORMANCE" in runner
    assert "_uncheckpointed_${run_started_utc}" in runner
    assert "WANDB_DISABLE_ARTIFACT=" in runner
    assert "--save_checkpoint_to_hub=" in trainer
    assert "nohup setsid env" in launcher
    assert "/etc/tmpfiles.d/iros-groot-n17.conf" in launcher
