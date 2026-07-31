"""Label Studio project の ep 別自動セットアップ + HF frame DL。

Epic #43 / Issue #44。

やること:
1. HF `Team-RAMEN/IROS2026_RAMEN_Hara_skillsplitframes_upperpolicy` から指定 ep の
   PNG を local `workspace/images/{skill}/ep{XX}/` に DL (既存 skip)
2. LS project を ep 別に作成 or 再利用 (title = `upperpolicy_ep{XX}`)
3. `labels_config.xml` (7-class OBB) を適用
4. LS Local storage を registrar + sync → 既存 PNG が LS UI で見える

Usage (LS は事前に `pixi run serve` で起動しっぱなし):
    pixi run bootstrap --ep 62
    pixi run bootstrap --eps 62,64,66
    pixi run bootstrap --ep-range 62-99

前提: `LABEL_STUDIO_USER_TOKEN` を設定して `pixi run serve` を起動していること。
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# ---- 定数 ----
LS_BASE = "http://localhost:8080"
LS_TOKEN = ""

HF_REPO = "Team-RAMEN/IROS2026_RAMEN_Hara_skillsplitframes_upperpolicy"

EXP_DIR = Path(__file__).parent.parent.resolve()  # data/annotation_tool/
LABEL_CONFIG_XML = EXP_DIR / "labels_config.xml"
LOCAL_FILES_ROOT = EXP_DIR / "workspace" / "images"  # dump_head_frames.py と同じ
HF_DL_STAGING = EXP_DIR / "workspace" / "hf_dl_staging"  # hf_hub_download の一時 landing

HEADERS: dict[str, str] = {}


def require_ls_token() -> None:
    global LS_TOKEN, HEADERS
    LS_TOKEN = os.environ.get("LABEL_STUDIO_USER_TOKEN", "")
    HEADERS = {"Authorization": f"Token {LS_TOKEN}"} if LS_TOKEN else {}
    if not LS_TOKEN:
        raise SystemExit("Set LABEL_STUDIO_USER_TOKEN before using the Label Studio API.")


# ------------------------------------------------------------
# LS API
# ------------------------------------------------------------
def wait_for_ls(timeout: int = 600) -> None:
    # LS 初回起動は Django DB migrations で 5-10 min かかる (workspace/ls_data/*.sqlite3
    # 再利用時は数十秒)。README の Troubleshooting 参照。
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


SKILL_NAMES = [
    "insert_table_leg", "move_to_table", "flip_table",
    "finalize_leg", "pick_table_leg", "move_table_base",
]


def get_or_create_local_storage(project_id: int, ep: int, images_root: Path) -> list[int]:
    """全 6 skill 分の LS Local Storage を per-skill 作成する。

    LS の regex_filter は file basename にのみ適用されるため、ep で絞るには
    `path` を per-skill/ep dir に直接向ける必要がある。

    Returns:
        作成 or 再利用した storage id list (6 個)。
    """
    r = requests.get(
        f"{LS_BASE}/api/storages/localfiles/", headers=HEADERS, params={"project": project_id}
    )
    r.raise_for_status()
    existing = r.json() if isinstance(r.json(), list) else r.json().get("results", [])
    existing_paths = {str(Path(s.get("path", "")).resolve()): s["id"] for s in existing}

    storage_ids: list[int] = []
    for skill in SKILL_NAMES:
        target = images_root / skill / f"ep{ep:04d}"
        if not target.exists():
            print(f"  [skip] no local dir {target}")
            continue
        if str(target.resolve()) in existing_paths:
            sid = existing_paths[str(target.resolve())]
            print(f"  reuse storage id={sid} skill={skill}")
            storage_ids.append(sid)
            continue
        r = requests.post(
            f"{LS_BASE}/api/storages/localfiles/",
            headers=HEADERS,
            json={
                "title": f"{skill}",
                "path": str(target.resolve()),
                "project": project_id,
                "regex_filter": r".*\.png$",
                "use_blob_urls": True,
                "recursive_scan": False,
            },
        )
        if r.status_code >= 400:
            raise SystemExit(f"failed to create local storage: {r.status_code} {r.text}")
        sid = r.json()["id"]
        print(f"  created storage id={sid} skill={skill}")
        storage_ids.append(sid)
    return storage_ids


def sync_storage(storage_id: int) -> None:
    r = requests.post(
        f"{LS_BASE}/api/storages/localfiles/{storage_id}/sync", headers=HEADERS
    )
    r.raise_for_status()
    print(f"  synced storage id={storage_id}")


# ------------------------------------------------------------
# HF DL
# ------------------------------------------------------------
def hf_files_for_ep(ep: int, token: str | None) -> list[str]:
    """HF repo から `frames/*/ep{ep:04d}/frame_*.png` を全部列挙。"""
    api = HfApi(token=token)
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    pat = re.compile(rf"frames/[^/]+/ep{ep:04d}/frame_\d+\.png$")
    return sorted(f for f in files if pat.match(f))


def _dl_one(hf_path: str, token: str | None) -> tuple[str, Path]:
    """1 file DL を staging に置いてから最終位置へ move。既存なら noop。"""
    local_rel = hf_path[len("frames/"):]  # strip "frames/" prefix
    local_dst = LOCAL_FILES_ROOT / local_rel
    if local_dst.exists():
        return hf_path, local_dst
    local_dst.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        filename=hf_path,
        local_dir=str(HF_DL_STAGING),
        token=token,
    )
    # downloaded = HF_DL_STAGING / hf_path → move to LOCAL_FILES_ROOT / local_rel
    shutil.move(downloaded, local_dst)
    return hf_path, local_dst


def dl_ep_frames_parallel(ep: int, hf_paths: list[str], token: str | None, workers: int = 8) -> int:
    """並列 DL。既存 skip。DL した数を返す。"""
    to_dl = [
        f for f in hf_paths if not (LOCAL_FILES_ROOT / f[len("frames/"):]).exists()
    ]
    print(f"  hf={len(hf_paths)} local={len(hf_paths) - len(to_dl)} to_dl={len(to_dl)}")
    if not to_dl:
        return 0
    HF_DL_STAGING.mkdir(parents=True, exist_ok=True)
    n_dl = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_dl_one, f, token) for f in to_dl]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            n_dl += 1
            if i % 50 == 0 or i == len(to_dl):
                print(f"    ...{i}/{len(to_dl)} DL 済")
    # staging cleanup (残った空 dir だけ、下手に消さない)
    return n_dl


# ------------------------------------------------------------
# ep arg parsing
# ------------------------------------------------------------
def resolve_eps(args: argparse.Namespace) -> list[int]:
    if args.ep is not None:
        return [args.ep]
    if args.eps:
        return sorted({int(x.strip()) for x in args.eps.split(",") if x.strip()})
    if args.ep_range:
        lo, hi = args.ep_range.split("-")
        return list(range(int(lo), int(hi) + 1))
    raise SystemExit("must specify --ep / --eps / --ep-range")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> None:
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path)
    require_ls_token()

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ep", type=int, help="単一 ep 番号 (e.g. 62)")
    g.add_argument("--eps", type=str, help="複数 ep, comma-separated (e.g. '62,64,66')")
    g.add_argument("--ep-range", type=str, help="ep 範囲 (e.g. '62-99')")
    p.add_argument("--skip-dl", action="store_true", help="HF DL を skip (local に既存 assumed)")
    p.add_argument("--workers", type=int, default=8, help="並列 DL worker 数")
    args = p.parse_args()

    if not LABEL_CONFIG_XML.exists():
        raise SystemExit(f"label config not found: {LABEL_CONFIG_XML}")
    label_config = LABEL_CONFIG_XML.read_text()

    eps = resolve_eps(args)
    print(f"[bootstrap] target eps: {eps}")
    print()

    wait_for_ls()
    print()

    token = None  # dotenv で HF_TOKEN が入ってる

    for ep in eps:
        print(f"=== ep {ep:04d} ===")
        # 1. HF DL (skip-dl でない場合のみ HF query + DL)。skip-dl なら local 前提。
        if not args.skip_dl:
            hf_paths = hf_files_for_ep(ep, token)
            if not hf_paths:
                print(f"  [skip] no frames on HF for ep {ep:04d}")
                continue
            dl_ep_frames_parallel(ep, hf_paths, token, workers=args.workers)
        else:
            # local に何も無ければ skip
            has_any_local = any(
                (LOCAL_FILES_ROOT / s / f"ep{ep:04d}").exists() for s in SKILL_NAMES
            )
            if not has_any_local:
                print(f"  [skip] no local frames for ep {ep:04d} (workspace/images/*/ep{ep:04d}/)")
                continue

        # 3. LS project 作成 or 再利用
        title = f"upperpolicy_ep{ep:04d}"
        desc = f"IROS 2026 Upper Policy annotation for episode {ep:04d} (HF: {HF_REPO})"
        pid = get_or_create_project(title, desc, label_config)

        # 4. Local storage 登録 (6 skill × 1 ep = 6 storages)
        sids = get_or_create_local_storage(pid, ep, LOCAL_FILES_ROOT)

        # 5. sync 全 storage
        for sid in sids:
            sync_storage(sid)

        url = f"{LS_BASE}/projects/{pid}/data"
        print(f"  URL: {url}")
        print()

    print("=" * 60)
    print("DONE. Open the URL(s) above in your browser.")
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
