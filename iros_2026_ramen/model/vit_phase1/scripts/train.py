"""ViT Phase 1 training CLI entry (Issue #18).

使い方:
    # smoke (10ep 相当、Home3 sample、CPU/小 GPU)
    pixi run -e train python -m model.vit_phase1.scripts.train \
        --config model/vit_phase1/configs/phase1.yaml \
        --lerobot-root ../research/data/lerobot_sample/Home3 \
        --labels-parquet data/vit_phase1/results/labels.parquet \
        --epochs 2 --batch-size 4 --num-workers 0 --run-name smoke_dev

    # 533ep 本走 (Sakura H100)
    pixi run -e train python -m model.vit_phase1.scripts.train \
        --config model/vit_phase1/configs/phase1.yaml \
        --run-name phase1_1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml
from dotenv import find_dotenv, load_dotenv
from torch.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

# --- pixi run から起動する想定なので、リポジトリ root が cwd の前提 ---
from evaluate.vit_phase1.metrics import promotion_flags
from model.vit_phase1.model.adapter import Adapter
from model.vit_phase1.model.backbone import DinoV3Backbone
from model.vit_phase1.model.heads import MultiHead
from model.vit_phase1.model.vit_module import Vit4HeadModel
from model.vit_phase1.train.dataset import Vit4HeadDataset, build_frameskip_weights
from model.vit_phase1.train.losses import Vit4HeadLoss
from model.vit_phase1.train.trainer import (
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)


# ==============================================================================
# Config loading + CLI override
# ==============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-name", type=str, default=None, help="wandb run name override")
    p.add_argument("--lerobot-root", type=Path, default=None)
    p.add_argument("--labels-parquet", type=Path, default=None)
    p.add_argument("--skill-durations-parquet", type=Path, default=None)
    p.add_argument(
        "--image-source",
        choices=["video", "jpg"],
        default=None,
        help="video (mp4 seek) or jpg (precomputed cache)",
    )
    p.add_argument("--jpg-root", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--grad-accum-steps", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cuda | cpu")
    p.add_argument("--no-wandb", action="store_true", help="wandb を無効化 (smoke 用)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="build 完了までで停止 (train loop 前に return)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="config の logging.checkpoint_dir override",
    )
    p.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="checkpoint (.pt) から model/optimizer/scheduler/epoch を復元して継続。"
             " `--resume-from auto` を渡すと ckpt_root/last.pt を自動探索",
    )
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg = yaml.safe_load(args.config.read_text())

    # CLI override 反映
    def _set_if(path: list, value):
        if value is None:
            return
        node = cfg
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value

    _set_if(["experiment", "run_name"], args.run_name)
    _set_if(["data", "lerobot_root"], str(args.lerobot_root) if args.lerobot_root else None)
    _set_if(["data", "labels_parquet"], str(args.labels_parquet) if args.labels_parquet else None)
    _set_if(
        ["data", "skill_durations_parquet"],
        str(args.skill_durations_parquet) if args.skill_durations_parquet else None,
    )
    _set_if(["data", "image_source"], args.image_source)
    _set_if(["data", "jpg_root"], str(args.jpg_root) if args.jpg_root else None)
    _set_if(["training", "epochs"], args.epochs)
    _set_if(["training", "batch_size"], args.batch_size)
    _set_if(["training", "num_workers"], args.num_workers)
    _set_if(["training", "grad_accum_steps"], args.grad_accum_steps)
    _set_if(["runtime", "device"], args.device)
    _set_if(["runtime", "seed"], args.seed)
    _set_if(["logging", "checkpoint_dir"], str(args.checkpoint_dir) if args.checkpoint_dir else None)

    return cfg


def _short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


# ==============================================================================
# Component builders
# ==============================================================================
def build_model(cfg: dict) -> Vit4HeadModel:
    bb_cfg = cfg["model"]["backbone"]
    ad_cfg = cfg["model"]["adapter"]
    hd_cfg = cfg["model"]["heads"]

    backbone = DinoV3Backbone(hf_id=bb_cfg["hf_id"])
    adapter = Adapter(
        in_dim=ad_cfg["in_dim"],
        out_dim=ad_cfg["out_dim"],
        dropout=ad_cfg.get("dropout", 0.0),
    )
    # v2 (Issue #28): heads.worldstate があれば 5 head 化。無ければ従来 4 head。
    ws_cfg = hd_cfg.get("worldstate")
    include_worldstate = ws_cfg is not None
    # v2 H5 (Issue #28): skill_condition が model 直下にある場合は conditional model
    cond_cfg = cfg["model"].get("skill_condition", {}) or {}
    skill_condition = bool(cond_cfg.get("enabled", False))
    skill_emb_dim = int(cond_cfg.get("emb_dim", 32))
    hint_noise_prob = float(cond_cfg.get("hint_noise_prob", 0.0))
    heads = MultiHead(
        in_dim=ad_cfg["out_dim"],
        hidden_dim=hd_cfg["skill"]["hidden_dim"],
        num_skills=hd_cfg["skill"]["out_dim"],
        include_worldstate=include_worldstate,
        num_legs=ws_cfg["out_dim"] if include_worldstate else 5,
        skill_condition=skill_condition,
        skill_emb_dim=skill_emb_dim,
    )
    # hint_noise_prob は heads の attribute として持たせる (trainer が読む)
    heads.hint_noise_prob = hint_noise_prob
    return Vit4HeadModel(backbone, adapter, heads)


def compute_worldstate_class_weight(
    labels_parquet: Path, num_legs: int, split: str = "train"
) -> torch.Tensor:
    """train split の num_legs_inserted 分布から inverse-frequency weight を計算.

    v2 (Issue #28)。skill_durations.parquet に相当する worldstate 用 stat を preprocess
    追加するとメンテコストが上がるので、trainer 起動時に labels.parquet を 1 回 scan する
    on-the-fly 方式を採用。
    """
    t = pq.read_table(labels_parquet)
    t = t.filter(pc.equal(t["split"], split))
    nli = t["num_legs_inserted"].to_numpy().astype(np.int64)
    total = nli.size
    weights = np.zeros(num_legs, dtype=np.float32)
    for c in range(num_legs):
        count = int((nli == c).sum())
        if count > 0:
            # inverse-frequency normalized to mean=1.0
            weights[c] = total / (num_legs * count)
    # 0 count のクラスは 1.0 (無害な default)
    weights[weights == 0] = 1.0
    return torch.from_numpy(weights)


def load_class_weight(
    skill_durations_parquet: Path,
    num_skills: int,
    mode: str = "inverse",
    clip: float | None = None,
) -> torch.Tensor:
    """skill_durations.parquet の class_weight 列を tensor 化.

    Pass2 (Issue #36): 完全逆頻度 (mode="inverse") は balanced 目的となり、支配
    クラス (finalize_leg 56%) を犠牲に希少クラス (move_table_base 0.8%) を過剰予測し、
    skill_top1 が majority baseline (0.56) を割り込む崩壊を招く (val 実測 0.37)。
    mode / clip で緩和する。

    Args:
        mode: "inverse" (現行=逆頻度そのまま) | "sqrt" (逆頻度の平方根で緩和) |
              "uniform" (全 valid skill を 1.0、= 素の top1 目的)。
        clip: 非 None なら nonzero 重みを上限 clip でクリップ。極端な希少クラス
              重みの暴走を抑える。mode 適用後に効く。drop skill (weight=0) は 0 維持。
    """
    t = pq.read_table(skill_durations_parquet)
    skill_ids = t["skill_id"].to_numpy()
    cw = t["class_weight"].to_numpy().astype(np.float32)
    weights = np.zeros(num_skills, dtype=np.float32)
    for sid, w in zip(skill_ids, cw):
        weights[int(sid)] = float(w)

    nonzero = weights > 0  # drop skill (weight=0) は変換対象外
    if mode == "uniform":
        weights[nonzero] = 1.0
    elif mode == "sqrt":
        weights[nonzero] = np.sqrt(weights[nonzero])
    elif mode != "inverse":
        raise ValueError(f"unknown class_weight_mode: {mode!r}")
    if clip is not None:
        weights[nonzero] = np.minimum(weights[nonzero], float(clip))
    return torch.from_numpy(weights)


def compute_anomaly_pos_weight(labels_parquet: Path, split: str = "train") -> float:
    """train split の visual_anomaly から neg/pos 比率を返す."""
    t = pq.read_table(labels_parquet)
    t = t.filter(pc.equal(t["split"], split))
    a = t["visual_anomaly"].to_numpy().astype(bool)
    pos = int(a.sum())
    neg = int((~a).sum())
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


def build_dataloaders(cfg: dict, backbone_mean, backbone_std) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    aug_cfg = cfg["augmentation"]

    common = dict(
        labels_parquet=data_cfg["labels_parquet"],
        lerobot_root=data_cfg["lerobot_root"],
        image_size=data_cfg["image_size"],
        normalize_mean=backbone_mean,
        normalize_std=backbone_std,
        image_source=data_cfg.get("image_source", "video"),
        jpg_root=data_cfg.get("jpg_root"),
    )

    train_ds = Vit4HeadDataset(
        split="train",
        photometric_aug=True,
        photometric_strength=aug_cfg["photometric"]["strength"],
        **common,
    )
    val_ds = Vit4HeadDataset(
        split="val",
        photometric_aug=False,
        **common,
    )

    # FrameSkip weighted sampling (train のみ)
    fs = aug_cfg["frameskip"]
    weights = build_frameskip_weights(
        data_cfg["labels_parquet"],
        split="train",
        retention=fs["retention"],
        gripper_boost=fs["gripper_transition_boost"],
    )
    train_sampler = WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        worker_init_fn=Vit4HeadDataset.worker_init_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        worker_init_fn=Vit4HeadDataset.worker_init_fn,
    )
    return train_dl, val_dl


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    args = parse_args()
    load_dotenv(find_dotenv(usecwd=True))
    cfg = resolve_config(args)

    # ---- run_name の解決 (未指定なら short_sha を挿入) ----
    if cfg["experiment"]["run_name"] in (None, "", "phase1_dev"):
        cfg["experiment"]["run_name"] = f"phase1_{_short_sha()}"
    run_name = cfg["experiment"]["run_name"]

    # ---- reproducibility ----
    seed = int(cfg["runtime"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- device ----
    dev_str = cfg["runtime"]["device"]
    if dev_str == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available, falling back to cpu", file=sys.stderr)
        dev_str = "cpu"
    device = torch.device(dev_str)

    # ---- wandb ----
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg["logging"]["wandb"]["project"],
                entity=cfg["logging"]["wandb"]["entity"],
                name=run_name,
                config=cfg,
            )
        except Exception as e:
            print(f"[warn] wandb init failed: {e}", file=sys.stderr)
            wandb_run = None

    # ---- model ----
    print(f"[build] backbone hf_id={cfg['model']['backbone']['hf_id']}")
    model = build_model(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[build] trainable={n_trainable:,} / total={n_total:,}")

    # ---- dataloaders ----
    train_dl, val_dl = build_dataloaders(
        cfg,
        backbone_mean=tuple(model.backbone.normalize_mean.flatten().tolist()),
        backbone_std=tuple(model.backbone.normalize_std.flatten().tolist()),
    )
    print(f"[build] train_frames={len(train_dl.dataset)} val_frames={len(val_dl.dataset)}")

    # ---- loss ----
    skill_loss_cfg = cfg["loss"].get("skill", {}) or {}
    cw_mode = str(skill_loss_cfg.get("class_weight_mode", "inverse"))
    cw_clip = skill_loss_cfg.get("class_weight_clip")
    class_weight = load_class_weight(
        Path(cfg["data"]["skill_durations_parquet"]),
        num_skills=cfg["data"]["num_skills"],
        mode=cw_mode,
        clip=cw_clip,
    )
    pos_weight = compute_anomaly_pos_weight(
        Path(cfg["data"]["labels_parquet"]), split="train"
    )
    # v2 (Issue #28): worldstate head 有効時のみ class_weight_worldstate を組み立て
    heads_cfg = cfg["model"]["heads"]
    ws_hd_cfg = heads_cfg.get("worldstate")
    if ws_hd_cfg is not None:
        num_legs = int(ws_hd_cfg["out_dim"])
        cw_worldstate = compute_worldstate_class_weight(
            Path(cfg["data"]["labels_parquet"]), num_legs=num_legs, split="train"
        )
        ws_loss_cfg = cfg["loss"].get("worldstate", {})
        ls_worldstate = float(ws_loss_cfg.get("label_smoothing", 0.1))
        print(f"[build] worldstate class_weight={cw_worldstate.tolist()} label_smoothing={ls_worldstate}")
    else:
        cw_worldstate = None
        ls_worldstate = 0.1  # 未使用

    # v2 H5 (Issue #28): worldstate loss で insert frame を boost する対応
    ws_loss_cfg = cfg["loss"].get("worldstate", {}) or {}
    insert_boost = float(ws_loss_cfg.get("insert_boost", 1.0))
    insert_skill_id = int(cfg["data"].get("insertion_skill_id", 0))
    # Pass2 (Issue #36): phase 教師末尾 (target >= threshold) の loss weight を boost
    phase_loss_cfg = cfg["loss"].get("phase", {}) or {}
    phase_tail_weight_ratio = float(phase_loss_cfg.get("tail_weight_ratio", 1.0))
    phase_tail_threshold = float(phase_loss_cfg.get("tail_threshold", 0.8))
    loss_fn = Vit4HeadLoss(
        class_weight=class_weight,
        pos_weight=pos_weight,
        weights=cfg["loss"]["weights"],
        label_smoothing=cfg["loss"]["skill"].get("label_smoothing", 0.0),
        class_weight_worldstate=cw_worldstate,
        label_smoothing_worldstate=ls_worldstate,
        worldstate_insert_boost=insert_boost,
        insert_skill_id=insert_skill_id,
        phase_tail_weight_ratio=phase_tail_weight_ratio,
        phase_tail_threshold=phase_tail_threshold,
    ).to(device)
    print(
        f"[build] class_weight={class_weight.tolist()} "
        f"(mode={cw_mode} clip={cw_clip}) pos_weight={pos_weight:.3f}"
    )

    # ---- optimizer + scheduler ----
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg["weight_decay"],
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
    )
    sch_cfg = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sch_cfg["T_max"]
    )
    scaler = GradScaler(device.type) if cfg["training"]["amp"] and device.type == "cuda" else None

    # ---- dry-run 早期 return ----
    if args.dry_run:
        print("[dry-run] build ok, exiting before training loop")
        if wandb_run:
            wandb_run.finish()
        return

    # ---- Training loop ----
    ckpt_root = Path(cfg["logging"]["checkpoint_dir"]) / run_name
    ckpt_root.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0  # skill_top1 で選ぶ、大きい方が良い
    start_epoch = 0

    # ---- Resume ----
    resume_path: Path | None = None
    if args.resume_from is not None:
        if str(args.resume_from) == "auto":
            candidate = ckpt_root / "last.pt"
            if candidate.exists():
                resume_path = candidate
            else:
                print(f"[resume] auto: no last.pt under {ckpt_root}, starting fresh")
        else:
            resume_path = args.resume_from
    if resume_path is not None:
        print(f"[resume] loading {resume_path}")
        info = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=device)
        # completed=True: その epoch の全 step 済 → 次 epoch から。
        # completed=False (mid-epoch save): その epoch を最初から再走 (data loader 状態は
        # 保存していないので、model+opt state だけ持ち込んで epoch 頭から)。
        # backward compat: 古い ckpt (completed 未保存) は True 扱いで従前挙動
        completed = bool(info["metrics"].get("completed", True))
        start_epoch = int(info["epoch"]) + (1 if completed else 0)
        best_metric = float(info["metrics"].get("skill_top1", best_metric))
        print(
            f"[resume] start_epoch={start_epoch} "
            f"(completed={completed}, at step={info['metrics'].get('step','?')}) "
            f"best_skill_top1={best_metric:.4f}"
        )

    val_thresholds = cfg["validation"]["metrics"]
    epochs = int(cfg["training"]["epochs"])
    log_every = int(cfg["training"].get("log_every_n_steps", 50))
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    predictable_skills = set(int(x) for x in cfg["data"]["predictable_skills"])

    save_every_n_steps = int(cfg["training"].get("save_every_n_steps", 1000))

    def _save_last(epoch: int, step: int | None, completed: bool):
        """mid-epoch (completed=False) と end-of-epoch (True) 両対応の last.pt 更新."""
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {
                "step": step if step is not None else -1,
                "completed": completed,
                **(val_metrics if completed else {}),
            },
            ckpt_root / "last.pt",
        )

    val_metrics: dict[str, float] = {}  # 0 epoch のみ回さず終了する resume ケース対策
    for epoch in range(start_epoch, epochs):
        print(f"\n=== epoch {epoch + 1}/{epochs} ===")
        train_metrics = train_one_epoch(
            model,
            loss_fn,
            train_dl,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            wandb_run=wandb_run,
            log_every_n_steps=log_every,
            grad_accum_steps=grad_accum,
            save_callback=_save_last,
            save_every_n_steps=save_every_n_steps,
        )
        scheduler.step()
        val_metrics = evaluate(
            model,
            val_dl,
            device,
            predictable_skills=predictable_skills,
            insertion_skill_id=int(cfg["data"].get("insertion_skill_id", 0)),
        )
        val_metrics["lr"] = optimizer.param_groups[0]["lr"]

        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}, indent=2))
        if wandb_run:
            wandb_run.log(
                {
                    **{f"train_avg/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "epoch": epoch,
                }
            )

        # Checkpoint
        # last.pt は毎 epoch 末尾で completed=True マークして上書き
        _save_last(epoch=epoch, step=None, completed=True)
        save_every = int(cfg["logging"]["save_every_n_epochs"])
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                ckpt_root / f"epoch_{epoch + 1:03d}.pt",
            )
        if val_metrics["skill_top1"] > best_metric:
            best_metric = val_metrics["skill_top1"]
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_metrics,
                ckpt_root / "best.pt",
            )

    # ---- 参考: 最終 epoch の Phase 2 promotion 判定 ----
    # 正式な Phase 2 promotion 判定は eval CLI (`evaluate.vit_phase1.run_eval`) で
    # 任意 checkpoint に対して行う (Issue #21)。train ループでは stdout の参考値のみ、
    # wandb summary には best_skill_top1 だけを流す (視認情報の一貫性維持)。
    passed = promotion_flags(val_metrics, val_thresholds)
    print("\n=== Phase 2 promotion check (last epoch val_metrics, reference only) ===")
    print(json.dumps(passed, indent=2))
    print(f"best_skill_top1={best_metric:.4f} (threshold {val_thresholds['skill_top1_threshold']})")
    print("For the authoritative promotion decision, run `pixi run -e train eval --checkpoint ...`.")

    if wandb_run:
        wandb_run.summary.update({"best_skill_top1": best_metric})
        wandb_run.finish()


if __name__ == "__main__":
    main()
