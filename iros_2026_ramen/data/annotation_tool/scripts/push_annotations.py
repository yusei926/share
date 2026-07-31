"""LS project の annotations を fetch → YOLO OBB 変換 → HF push まで一発。

Epic #43 / Issue #44。

処理:
1. LS API から指定 ep の project の annotations JSON を取得
2. `workspace/exports/ep{XX}_{timestamp}.json` に raw を保存 (timestamped、上書きせず)
3. `ls_export_to_yolo_obb.py` の関数で YOLO OBB txt に変換 →
   `workspace/yolo_obb/{skill}__ep{XX}__frame_{YY}.txt`
4. HF `Team-RAMEN/IROS2026_RAMEN_Hara_upperpolicy_annotations` に batch commit で push:
   - `ls_exports/ep{XX}_{timestamp}.json` (source of truth、追記)
   - `yolo_obb/labels/*.txt` (training ready、上書き)

Usage (LS 起動中で annotation 済み前提):
    pixi run push-annotations --ep 60
    pixi run push-annotations --eps 60,62
    pixi run push-annotations --ep 60 --no-hf-upload  # local 変換のみ

    # v2 semi-supervised (Issue #53) のような custom title project は --project-id 指定:
    pixi run push-annotations --ep 62 --project-id 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, create_repo

# scripts/ 配下からの import
from ls_export_to_yolo_obb import parse_ls_export, write_yolo_obb_txts

# ---- 定数 ----
LS_BASE = "http://localhost:8080"
LS_TOKEN = ""
HEADERS: dict[str, str] = {}


def require_ls_token() -> None:
    global LS_TOKEN, HEADERS
    LS_TOKEN = os.environ.get("LABEL_STUDIO_USER_TOKEN", "")
    HEADERS = {"Authorization": f"Token {LS_TOKEN}"} if LS_TOKEN else {}
    if not LS_TOKEN:
        raise SystemExit("Set LABEL_STUDIO_USER_TOKEN before using the Label Studio API.")

HF_DEST_REPO = "Team-RAMEN/IROS2026_RAMEN_Hara_upperpolicy_annotations"

EXP_DIR = Path(__file__).parent.parent.resolve()
EXPORT_DIR = EXP_DIR / "workspace" / "exports"
YOLO_OUT_DIR = EXP_DIR / "workspace" / "yolo_obb"


# ------------------------------------------------------------
# LS API
# ------------------------------------------------------------
def find_project_by_ep(ep: int) -> int | None:
    """title=`upperpolicy_ep{XX:04d}` の project id を返す (無ければ None)。"""
    title = f"upperpolicy_ep{ep:04d}"
    r = requests.get(f"{LS_BASE}/api/projects/", headers=HEADERS)
    r.raise_for_status()
    payload = r.json()
    projects = payload if isinstance(payload, list) else payload.get("results", [])
    for p in projects:
        if p.get("title") == title:
            return p["id"]
    return None


def fetch_ls_export(project_id: int) -> list[dict]:
    """LS の export API から full JSON export を取得 (task + annotations)。"""
    r = requests.get(
        f"{LS_BASE}/api/projects/{project_id}/export",
        headers=HEADERS,
        params={"exportType": "JSON"},
    )
    r.raise_for_status()
    return r.json()


# ------------------------------------------------------------
# HF push
# ------------------------------------------------------------
def ensure_dest_repo(token: str | None) -> None:
    create_repo(
        repo_id=HF_DEST_REPO,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )


def push_to_hf(
    ls_export_files: dict[Path, str],
    yolo_txt_files: dict[Path, str],
    token: str | None,
) -> None:
    """batch commit で ls_exports/*.json + yolo_obb/labels/*.txt を push。

    Args:
        ls_export_files: {local_path: remote_relpath (under repo)}
        yolo_txt_files: {local_path: remote_relpath}
    """
    api = HfApi(token=token)
    ops = []
    for local, remote in {**ls_export_files, **yolo_txt_files}.items():
        ops.append(CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local)))
    if not ops:
        print("[hf-up] nothing to push")
        return
    api.create_commit(
        repo_id=HF_DEST_REPO,
        repo_type="dataset",
        operations=ops,
        commit_message=f"push {len(ls_export_files)} ls_exports + {len(yolo_txt_files)} yolo_obb txts",
        token=token,
    )
    print(f"[hf-up] pushed {len(ops)} files → {HF_DEST_REPO}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def resolve_eps(args: argparse.Namespace) -> list[int]:
    if args.ep is not None:
        return [args.ep]
    if args.eps:
        return sorted({int(x.strip()) for x in args.eps.split(",") if x.strip()})
    raise SystemExit("must specify --ep or --eps")


def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
    require_ls_token()

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ep", type=int, help="単一 ep 番号 (e.g. 60)")
    g.add_argument("--eps", type=str, help="複数 ep, comma-separated (e.g. '60,62')")
    p.add_argument(
        "--project-id",
        type=int,
        default=None,
        help=(
            "LS project id (ep-based title lookup を bypass)。"
            "custom title の project (v2 semi-supervised 等) 用。--ep 単一時のみ有効。"
        ),
    )
    p.add_argument(
        "--no-hf-upload",
        dest="hf_upload",
        action="store_false",
        help="HF push を skip (local export + 変換のみ)",
    )
    p.set_defaults(hf_upload=True)
    args = p.parse_args()

    eps = resolve_eps(args)
    if args.project_id is not None and len(eps) != 1:
        raise SystemExit("--project-id は --ep 単一時のみ有効 (--eps 複数指定と併用不可)")
    print(f"[push] target eps: {eps}")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    YOLO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    token = None  # dotenv 経由

    if args.hf_upload:
        print(f"[hf] ensuring {HF_DEST_REPO}")
        ensure_dest_repo(token)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_ls_files: dict[Path, str] = {}
    all_yolo_files: dict[Path, str] = {}

    for ep in eps:
        print(f"=== ep {ep:04d} ===")

        # 1. LS project 探す (--project-id 指定時はそちらを優先)
        if args.project_id is not None:
            pid = args.project_id
            print(f"  project_id={pid} (from --project-id)")
        else:
            pid = find_project_by_ep(ep)
            if pid is None:
                print(f"  [skip] no LS project for ep {ep:04d}")
                continue
            print(f"  project_id={pid}")

        # 2. LS export fetch
        export = fetch_ls_export(pid)
        if not export:
            print(f"  [skip] LS export empty")
            continue
        ann_count = sum(len(t.get("annotations", [])) for t in export)
        box_count = sum(
            len(a.get("result", []))
            for t in export
            for a in t.get("annotations", [])
        )
        print(f"  tasks={len(export)} annotations={ann_count} boxes={box_count}")

        # 3. LS raw JSON 保存 (timestamped)
        ls_json_path = EXPORT_DIR / f"ep{ep:04d}_{timestamp}.json"
        with open(ls_json_path, "w") as f:
            json.dump(export, f)
        all_ls_files[ls_json_path] = f"ls_exports/{ls_json_path.name}"

        # 4. YOLO OBB txt 変換 (hierarchical: {skill}/ep{XX}/frame_YY.txt)
        parsed = parse_ls_export(ls_json_path)
        n_txt = write_yolo_obb_txts(parsed, YOLO_OUT_DIR)
        print(f"  wrote {n_txt} txt → {YOLO_OUT_DIR}")
        # HF は yolo_obb/labels/{rel_path}.txt に置く (parsed の key = rel_path)
        for rel_path in parsed:
            local_txt = YOLO_OUT_DIR / f"{rel_path}.txt"
            all_yolo_files[local_txt] = f"yolo_obb/labels/{rel_path}.txt"

    # 5. HF push (batch)
    if args.hf_upload:
        push_to_hf(all_ls_files, all_yolo_files, token)

    print(f"[done] ls_exports={len(all_ls_files)} yolo_txt={len(all_yolo_files)}")


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
