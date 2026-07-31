"""m_lowaug で ep 62 frame を pre-predict → LS project に predictions 付き task import。

Epic #43 / Issue #53 (v2 semi-supervised active learning cycle)。

「描くのではなく修正のみ」で annotation 工数 1/3~1/5 削減を狙う。
bootstrap_project.py の Local Storage 経由ではなく、tasks を明示 import して
predictions field を付与する。

Usage (LS は事前に `pixi run serve` で起動、YOLO weight は model/yolo_obb env で pixi 済):
    # model/yolo_obb sub-workspace の env で叩く (ultralytics 込み)
    cd model/yolo_obb
    pixi run python ../../data/annotation_tool/scripts/predict_and_prelabel.py \\
        --ep 62 --per-skill 50 --seed 42 \\
        --weight runs/m_lowaug/weights/best.pt \\
        --conf 0.25 \\
        --project-title upperpolicy_v2_ep0062_prelabel

前提:
- `data/annotation_tool/workspace/images/{skill}/ep0062/frame_*.png` が local に存在
  (annotation_tool 側の bootstrap で HF から DL 済)
- LS DOCUMENT_ROOT = `data/annotation_tool/workspace/` (annotation_tool の serve task)
- LS token はローカルの `LABEL_STUDIO_USER_TOKEN` から取得する

env について:
- annotation_tool env には ultralytics が無いため、model/yolo_obb sub-workspace を使う
- LS API は `requests` (HF hub 経由の transitive) のみで済み、label-studio-sdk 不要
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import uuid
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

# 同 dir の ls_export_to_yolo_obb から convert 関数と class map を再利用
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from ls_export_to_yolo_obb import (  # noqa: E402
    CLASS_ID_TO_NAME,
    yolo_obb_verts_to_ls_rect,
)
from bootstrap_project import (  # noqa: E402
    get_or_create_local_storage,
    require_ls_token as require_bootstrap_ls_token,
)

# ---- 定数 (bootstrap_project.py と揃える) ----
LS_BASE = "http://localhost:8080"
LS_TOKEN = ""
HEADERS: dict[str, str] = {}


def require_ls_token() -> None:
    global LS_TOKEN, HEADERS
    LS_TOKEN = os.environ.get("LABEL_STUDIO_USER_TOKEN", "")
    HEADERS = {"Authorization": f"Token {LS_TOKEN}"} if LS_TOKEN else {}
    if not LS_TOKEN:
        raise SystemExit("Set LABEL_STUDIO_USER_TOKEN before using the Label Studio API.")

EXP_DIR = Path(__file__).parent.parent.resolve()  # data/annotation_tool/
LABEL_CONFIG_XML = EXP_DIR / "labels_config.xml"
LOCAL_FILES_ROOT = EXP_DIR / "workspace" / "images"

# LS DOCUMENT_ROOT = workspace/ なので URL は `?d=images/...` になる
LS_URL_PREFIX = "/data/local-files/?d=images"

SKILL_NAMES = [
    "insert_table_leg", "move_to_table", "flip_table",
    "finalize_leg", "pick_table_leg", "move_table_base",
]


# ------------------------------------------------------------
# LS API
# ------------------------------------------------------------
def wait_for_ls(timeout: int = 600) -> None:
    print(f"waiting for LS at {LS_BASE} ...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{LS_BASE}/health/", timeout=2)
            if r.status_code == 200:
                print(" up.")
                return
        except requests.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    raise SystemExit(f"\nLS didn't come up within {timeout}s. Run `pixi run serve` first?")


def get_or_create_project(title: str, description: str, label_config: str) -> int:
    r = requests.get(
        f"{LS_BASE}/api/projects/", headers=HEADERS, params={"title": title}
    )
    r.raise_for_status()
    results = r.json().get("results") or r.json()
    if isinstance(results, list):
        for p in results:
            if p.get("title") == title:
                pid = p["id"]
                if (p.get("label_config") or "").strip() != label_config.strip():
                    rp = requests.patch(
                        f"{LS_BASE}/api/projects/{pid}/",
                        headers=HEADERS,
                        json={"label_config": label_config, "description": description},
                    )
                    rp.raise_for_status()
                    print(f"  reuse project id={pid} title='{title}' (label_config PATCHED)")
                else:
                    print(f"  reuse project id={pid} title='{title}' (label_config in sync)")
                return pid
    r = requests.post(
        f"{LS_BASE}/api/projects/",
        headers=HEADERS,
        json={"title": title, "description": description, "label_config": label_config},
    )
    r.raise_for_status()
    pid = r.json()["id"]
    print(f"  created project id={pid} title='{title}'")
    return pid


def import_tasks(project_id: int, tasks: list[dict]) -> int:
    """LS import endpoint で task list を一括 import。predictions は task 内に含める。"""
    r = requests.post(
        f"{LS_BASE}/api/projects/{project_id}/import",
        headers={**HEADERS, "Content-Type": "application/json"},
        json=tasks,
    )
    if r.status_code >= 400:
        raise SystemExit(f"import failed: {r.status_code} {r.text}")
    result = r.json() if r.content else {}
    n_imported = result.get("task_count", len(tasks))
    print(f"  imported {n_imported} tasks")
    return int(n_imported)


# ------------------------------------------------------------
# Frame sampling
# ------------------------------------------------------------
def sample_frames(ep: int, per_skill: int, seed: int) -> list[tuple[str, Path, str]]:
    """各 skill から seed 固定の非復元 sample。

    Returns:
        [(skill, abs_path, rel_path_from_LOCAL_FILES_ROOT), ...]
        rel_path 例: "insert_table_leg/ep0062/frame_01234.png"
    """
    rng = random.Random(seed)
    picks: list[tuple[str, Path, str]] = []
    for skill in SKILL_NAMES:
        skill_dir = LOCAL_FILES_ROOT / skill / f"ep{ep:04d}"
        if not skill_dir.is_dir():
            print(f"  [skip] no dir {skill_dir}")
            continue
        files = sorted(skill_dir.glob("frame_*.png"))
        n_avail = len(files)
        if n_avail < per_skill:
            print(f"  [warn] skill={skill} has {n_avail} < requested {per_skill} — using all")
            chosen = files
        else:
            chosen = rng.sample(files, per_skill)
        for f in chosen:
            picks.append((skill, f, str(f.relative_to(LOCAL_FILES_ROOT))))
        print(f"  sampled {len(chosen)}/{n_avail} from skill={skill}")
    return picks


# ------------------------------------------------------------
# YOLO OBB → LS prediction 変換 (pure、test 可能)
# ------------------------------------------------------------
def verts_px_to_ls_result(
    class_id: int,
    verts_px: list[tuple[float, float]],
    img_w: int,
    img_h: int,
) -> dict:
    """Ultralytics OBB corners (pixel、TL→TR→BR→BL 順) → LS RectangleLabels result 1 件。

    座標変換 (Issue #53 の aspect distortion fix 適用済):
    - pixel → %: (px/img_w*100, py/img_h*100)、x/y 軸で独立正規化
    - %-space 4 頂点から yolo_obb_verts_to_ls_rect (Theory C 版) で LS 値を復元。
      内部で pixel-space に戻して辺長 / rotation を計算するため、非正方 image でも
      LS UI と bit-perfect 一致する。
    """
    verts_pct = [
        (float(v[0]) / img_w * 100.0, float(v[1]) / img_h * 100.0) for v in verts_px
    ]
    ls_value = yolo_obb_verts_to_ls_rect(verts_pct, img_w, img_h)
    ls_value["rectanglelabels"] = [CLASS_ID_TO_NAME[class_id]]
    return {
        "id": uuid.uuid4().hex[:10],
        "type": "rectanglelabels",
        "from_name": "label",
        "to_name": "image",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": ls_value,
    }


def _strip_images_prefix(rel_path: str) -> str:
    """rel_path が 'images/' で始まっていたら strip (LS URL 側で prefix を付けるため)。"""
    if rel_path.startswith("images/"):
        return rel_path[len("images/"):]
    return rel_path


def build_ls_task(
    rel_path: str,
    boxes: list[dict],
    img_w: int,
    img_h: int,
    model_version: str,
) -> dict:
    """LS task 1 件を組み立てる。boxes = [{"class_id", "verts_px", "conf"}, ...]。

    boxes 空でも task は成立 (何も検出されなかった image、user が全部描く)。
    """
    task: dict = {
        "data": {"image": f"{LS_URL_PREFIX}/{_strip_images_prefix(rel_path)}"},
    }
    if not boxes:
        return task
    results = [
        verts_px_to_ls_result(b["class_id"], b["verts_px"], img_w, img_h) for b in boxes
    ]
    score = float(sum(b["conf"] for b in boxes) / len(boxes))
    task["predictions"] = [
        {
            "model_version": model_version,
            "score": score,
            "result": results,
        }
    ]
    return task


# ------------------------------------------------------------
# YOLO predict
# ------------------------------------------------------------
def run_yolo_predict(
    weight_path: Path, image_paths: list[Path], conf: float
) -> list[dict]:
    """Ultralytics YOLO で batch predict → 各 image の box list。

    Returns:
        [{"path": Path, "boxes": [{"class_id", "verts_px", "conf"}, ...],
          "img_w": int, "img_h": int}, ...]
    """
    from ultralytics import YOLO  # lazy import: root train env でのみ使える

    model = YOLO(str(weight_path))
    print(f"  running predict on {len(image_paths)} images (conf={conf}) ...")
    # Python 側で 1 image ずつ iterate (ultralytics の stream=True は場合により
    # 内部 batching を回避しきれず 8GB VRAM で OOM するため確実な逐次実行に統一)
    out = []
    for i, path in enumerate(image_paths, 1):
        if i % 50 == 0 or i == 1 or i == len(image_paths):
            print(f"    ...{i}/{len(image_paths)} predicted")
        res = model.predict(source=str(path), conf=conf, verbose=False)[0]
        img_h, img_w = res.orig_shape  # (H, W)
        entry = {
            "path": path,
            "boxes": [],
            "img_w": int(img_w),
            "img_h": int(img_h),
        }
        if res.obb is not None and len(res.obb) > 0:
            verts_all = res.obb.xyxyxyxy.cpu().numpy()  # (N, 4, 2)
            confs = res.obb.conf.cpu().numpy()
            clses = res.obb.cls.cpu().numpy().astype(int)
            for j in range(len(clses)):
                # Ultralytics xywhr2xyxyxyxy の出力順は local [BR, TR, TL, BL]
                # (ops.py: pt1 = center + w_vec + h_vec = BR, ...)。
                # yolo_obb_verts_to_ls_rect は [TL, TR, BR, BL] 前提なので入替え。
                v = verts_all[j]
                verts_canonical = [
                    (float(v[2][0]), float(v[2][1])),  # TL
                    (float(v[1][0]), float(v[1][1])),  # TR
                    (float(v[0][0]), float(v[0][1])),  # BR
                    (float(v[3][0]), float(v[3][1])),  # BL
                ]
                entry["boxes"].append(
                    {
                        "class_id": int(clses[j]),
                        "verts_px": verts_canonical,
                        "conf": float(confs[j]),
                    }
                )
        out.append(entry)
    n_boxes = sum(len(e["boxes"]) for e in out)
    print(f"  {n_boxes} total boxes across {len(out)} images")
    return out


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
    require_ls_token()
    require_bootstrap_ls_token()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ep", type=int, required=True, help="target episode (e.g. 62)")
    p.add_argument("--per-skill", type=int, default=50, help="frames per skill")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--weight", type=Path, required=True, help="YOLO OBB weight (best.pt)"
    )
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--project-title", type=str, required=True)
    p.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="LS prediction model_version (default: weight file の run 名)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="predict のみ、LS import しない"
    )
    args = p.parse_args()

    if not args.weight.exists():
        raise SystemExit(f"weight not found: {args.weight}")
    if not LABEL_CONFIG_XML.exists():
        raise SystemExit(f"label config not found: {LABEL_CONFIG_XML}")

    # runs/m_lowaug/weights/best.pt → m_lowaug
    model_version = args.model_version or args.weight.parent.parent.name

    print(
        f"[predict_and_prelabel] ep={args.ep} per_skill={args.per_skill} seed={args.seed}"
    )
    print(f"  weight={args.weight} model_version={model_version}")
    print()

    print("=== 1. sample frames ===")
    picks = sample_frames(args.ep, args.per_skill, args.seed)
    print(f"  total sampled: {len(picks)}")
    if not picks:
        raise SystemExit("no frames sampled — check LOCAL_FILES_ROOT")
    print()

    print("=== 2. YOLO predict ===")
    predictions = run_yolo_predict(args.weight, [p_[1] for p_ in picks], args.conf)

    tasks: list[dict] = []
    for (_skill, _abs_path, rel_path), pred in zip(picks, predictions):
        task = build_ls_task(
            rel_path, pred["boxes"], pred["img_w"], pred["img_h"], model_version
        )
        tasks.append(task)
    print()

    if args.dry_run:
        print("=== 3. dry-run summary ===")
        n_with_boxes = sum(1 for t in tasks if "predictions" in t)
        n_boxes = sum(
            len(t["predictions"][0]["result"]) for t in tasks if "predictions" in t
        )
        print(f"  {n_with_boxes}/{len(tasks)} tasks have predictions ({n_boxes} boxes)")
        import json

        sample = tasks[0] if tasks else {}
        print(f"  sample task[0]:\n{json.dumps(sample, indent=2)[:800]}")
        return

    print("=== 3. LS import ===")
    wait_for_ls()
    label_config = LABEL_CONFIG_XML.read_text()
    title = args.project_title
    desc = (
        f"IROS 2026 Upper Policy v2 pre-annotation for ep {args.ep:04d} "
        f"({model_version}, {len(tasks)} tasks)"
    )
    pid = get_or_create_project(title, desc, label_config)
    # Per-project Local Storage 登録 (LS の /data/local-files/?d= URL は
    # ACL として per-project storage を要求する)。sync は呼ばない — tasks は
    # import_tasks で明示投入するため storage sync による重複を避ける
    get_or_create_local_storage(pid, args.ep, LOCAL_FILES_ROOT)
    import_tasks(pid, tasks)
    print()
    print("=" * 60)
    print(f"DONE. URL: {LS_BASE}/projects/{pid}/data")
    print("login: use the local LABEL_STUDIO_USERNAME / LABEL_STUDIO_PASSWORD values")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        print(
            f"response: {e.response.text if e.response is not None else 'n/a'}",
            file=sys.stderr,
        )
        sys.exit(1)
