#!/usr/bin/env python3
"""Compare simulated wrist-camera frames against real Dex1+D405 references.

The score is intentionally simple and inspectable.  It does not replace a real
hand-eye calibration, but it gives a repeatable gate for the most visible
flip-table camera failures: too much distant floor/background, too much white
table underside, or too little black workbench/finger foreground.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


WRIST_ROLES = ("left_wrist", "right_wrist")
ROLE_FILENAMES = {
    "left_wrist": ("left_wrist_rgb.png", "left_wrist_real_f0010.png"),
    "right_wrist": ("right_wrist_rgb.png", "right_wrist_real_f0010.png"),
}
REPORT_VERSION = 2


@dataclass(frozen=True)
class ImageMetrics:
    path: str
    width: int
    height: int
    near_black_fraction: float
    dark_fraction: float
    white_fraction: float
    other_fraction: float
    top_near_black_fraction: float
    top_dark_fraction: float
    top_white_fraction: float
    top_other_fraction: float
    middle_near_black_fraction: float
    middle_dark_fraction: float
    middle_white_fraction: float
    middle_other_fraction: float
    bottom_near_black_fraction: float
    bottom_dark_fraction: float
    bottom_white_fraction: float
    bottom_other_fraction: float
    white_centroid_x: float | None
    white_centroid_y: float | None
    dark_centroid_x: float | None
    dark_centroid_y: float | None


@dataclass(frozen=True)
class RoleScore:
    role: str
    real: ImageMetrics
    sim: ImageMetrics
    score: float
    deltas: dict[str, float]


@dataclass(frozen=True)
class RealSample:
    label: str
    root: str
    paths: dict[str, str]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32)


def _safe_centroid(mask: np.ndarray) -> tuple[float | None, float | None]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    height, width = mask.shape
    return float(xs.mean() / max(width - 1, 1)), float(ys.mean() / max(height - 1, 1))


def _region_fraction(mask: np.ndarray, y0: int, y1: int) -> float:
    region = mask[y0:y1, :]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def image_metrics(path: Path) -> ImageMetrics:
    rgb = _load_rgb(path)
    height, width = rgb.shape[:2]
    luma = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)

    # The real flip-table wrist images are dominated by black Dex1 fingers,
    # black padded workbench, and the white IKEA table edge/underside.  These
    # thresholds are deliberately broad so lighting changes do not dominate.
    near_black = luma < 12.0
    dark = luma < 80.0
    white = (luma > 170.0) & (chroma < 70.0)
    other = ~(dark | white)

    top = height // 3
    middle = 2 * height // 3
    white_x, white_y = _safe_centroid(white)
    dark_x, dark_y = _safe_centroid(dark)

    return ImageMetrics(
        path=str(path),
        width=width,
        height=height,
        near_black_fraction=float(near_black.mean()),
        dark_fraction=float(dark.mean()),
        white_fraction=float(white.mean()),
        other_fraction=float(other.mean()),
        top_near_black_fraction=_region_fraction(near_black, 0, top),
        top_dark_fraction=_region_fraction(dark, 0, top),
        top_white_fraction=_region_fraction(white, 0, top),
        top_other_fraction=_region_fraction(other, 0, top),
        middle_near_black_fraction=_region_fraction(near_black, top, middle),
        middle_dark_fraction=_region_fraction(dark, top, middle),
        middle_white_fraction=_region_fraction(white, top, middle),
        middle_other_fraction=_region_fraction(other, top, middle),
        bottom_near_black_fraction=_region_fraction(near_black, middle, height),
        bottom_dark_fraction=_region_fraction(dark, middle, height),
        bottom_white_fraction=_region_fraction(white, middle, height),
        bottom_other_fraction=_region_fraction(other, middle, height),
        white_centroid_x=white_x,
        white_centroid_y=white_y,
        dark_centroid_x=dark_x,
        dark_centroid_y=dark_y,
    )


def _metric_value(metrics: ImageMetrics, key: str) -> float:
    value = getattr(metrics, key)
    if value is None:
        return 0.0
    return float(value)


def score_pair(real: ImageMetrics, sim: ImageMetrics) -> tuple[float, dict[str, float]]:
    keys = (
        "near_black_fraction",
        "dark_fraction",
        "white_fraction",
        "other_fraction",
        "top_near_black_fraction",
        "top_white_fraction",
        "top_other_fraction",
        "middle_near_black_fraction",
        "middle_dark_fraction",
        "middle_white_fraction",
        "bottom_near_black_fraction",
        "bottom_dark_fraction",
        "bottom_white_fraction",
        "white_centroid_y",
        "dark_centroid_y",
    )
    weights = {
        "near_black_fraction": 0.8,
        "dark_fraction": 1.0,
        "white_fraction": 1.0,
        "other_fraction": 0.8,
        "top_near_black_fraction": 0.6,
        "top_white_fraction": 0.9,
        "top_other_fraction": 1.2,
        "middle_near_black_fraction": 0.7,
        "middle_dark_fraction": 1.0,
        "middle_white_fraction": 1.0,
        "bottom_near_black_fraction": 1.1,
        "bottom_dark_fraction": 1.2,
        "bottom_white_fraction": 1.0,
        "white_centroid_y": 0.9,
        "dark_centroid_y": 0.7,
    }
    deltas: dict[str, float] = {}
    weighted = 0.0
    total_weight = 0.0
    for key in keys:
        delta = abs(_metric_value(sim, key) - _metric_value(real, key))
        deltas[key] = delta
        weighted += weights[key] * delta
        total_weight += weights[key]
    return weighted / total_weight, deltas


def _find_role_image(root: Path, role: str, *, recursive: bool = True) -> Path | None:
    names = ROLE_FILENAMES[role]
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    if not recursive:
        return None
    for name in names:
        matches = sorted(root.rglob(name))
        if matches:
            return matches[0]
    return None


def _real_image_key(path: Path, role: str) -> str | None:
    suffix = ".png"
    if not path.name.endswith(suffix):
        return None
    stem = path.name[: -len(suffix)]
    prefixes = (f"{role}_real_", f"{role}_")
    for prefix in prefixes:
        if stem.startswith(prefix):
            key = stem[len(prefix) :]
            if key in {"contact"} or key.endswith("_contact"):
                return None
            return key
    return None


def _discover_real_samples(root: Path) -> list[RealSample]:
    by_role: dict[str, dict[str, Path]] = {role: {} for role in WRIST_ROLES}
    for role in WRIST_ROLES:
        for path in sorted(root.rglob(f"{role}*.png")):
            key = _real_image_key(path, role)
            if key is None:
                continue
            by_role[role].setdefault(key, path)

    common_keys = sorted(set.intersection(*(set(paths) for paths in by_role.values())))
    samples: list[RealSample] = []
    for key in common_keys:
        role_paths = {role: str(by_role[role][key]) for role in WRIST_ROLES}
        label = root.name if key == "rgb" else f"{root.name}:{key}"
        samples.append(RealSample(label=label, root=str(root), paths=role_paths))

    if samples:
        return samples

    direct_paths: dict[str, str] = {}
    for role in WRIST_ROLES:
        path = _find_role_image(root, role)
        if path is None:
            return []
        direct_paths[role] = str(path)
    return [RealSample(label=root.name, root=str(root), paths=direct_paths)]


def _iter_sim_frame_dirs(root: Path) -> Iterable[Path]:
    if all(_find_role_image(root, role, recursive=False) is not None for role in WRIST_ROLES):
        yield root
        return

    for frame_dir in sorted(root.rglob("frame_*")):
        if all(_find_role_image(frame_dir, role, recursive=False) is not None for role in WRIST_ROLES):
            yield frame_dir


def _as_real_dirs(real_dirs: Path | list[Path]) -> list[Path]:
    if isinstance(real_dirs, Path):
        return [real_dirs]
    return real_dirs


def evaluate(real_dirs: Path | list[Path], sim_roots: list[Path], rank_by: str = "nearest") -> dict[str, object]:
    real_samples: list[RealSample] = []
    for real_dir in _as_real_dirs(real_dirs):
        real_samples.extend(_discover_real_samples(real_dir))
    if not real_samples:
        raise FileNotFoundError(f"Could not find paired wrist reference images in {real_dirs}")

    real_metrics_by_sample: dict[str, dict[str, ImageMetrics]] = {}
    for sample in real_samples:
        real_metrics_by_sample[sample.label] = {
            role: image_metrics(Path(path)) for role, path in sample.paths.items()
        }

    candidates: list[dict[str, object]] = []
    for sim_root in sim_roots:
        for frame_dir in _iter_sim_frame_dirs(sim_root):
            sample_scores: list[dict[str, object]] = []
            sim_metrics_by_role: dict[str, ImageMetrics] = {}
            for role in WRIST_ROLES:
                sim_path = _find_role_image(frame_dir, role, recursive=False)
                if sim_path is None:
                    raise FileNotFoundError(f"Could not find {role} image in {frame_dir}")
                sim_metrics_by_role[role] = image_metrics(sim_path)

            for sample in real_samples:
                role_scores: list[RoleScore] = []
                for role in WRIST_ROLES:
                    score, deltas = score_pair(
                        real_metrics_by_sample[sample.label][role],
                        sim_metrics_by_role[role],
                    )
                    role_scores.append(
                        RoleScore(
                            role=role,
                            real=real_metrics_by_sample[sample.label][role],
                            sim=sim_metrics_by_role[role],
                            score=score,
                            deltas=deltas,
                        )
                    )
                sample_scores.append(
                    {
                        "real_label": sample.label,
                        "real_paths": sample.paths,
                        "mean_score": float(np.mean([role_score.score for role_score in role_scores])),
                        "roles": [asdict(role_score) for role_score in role_scores],
                    }
                )
            sample_scores.sort(key=lambda item: float(item["mean_score"]))
            nearest_score = float(sample_scores[0]["mean_score"])
            mean_reference_score = float(np.mean([float(item["mean_score"]) for item in sample_scores]))
            candidates.append(
                {
                    "frame_dir": str(frame_dir),
                    "nearest_score": nearest_score,
                    "mean_reference_score": mean_reference_score,
                    "mean_score": nearest_score,
                    "nearest_real_label": sample_scores[0]["real_label"],
                    "sample_scores": sample_scores,
                    "roles": sample_scores[0]["roles"],
                }
            )

    rank_key = "mean_score" if rank_by == "nearest" else "mean_reference_score"
    candidates.sort(key=lambda item: float(item[rank_key]))
    return {
        "version": REPORT_VERSION,
        "rank_by": rank_by,
        "real_dirs": [str(path) for path in _as_real_dirs(real_dirs)],
        "real_dir": str(_as_real_dirs(real_dirs)[0]),
        "real_samples": [asdict(sample) for sample in real_samples],
        "sim_roots": [str(root) for root in sim_roots],
        "best_frame_dir": candidates[0]["frame_dir"] if candidates else None,
        "candidates": candidates,
    }


def _thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        thumb = image.convert("RGB")
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - thumb.width) // 2
    y = (size[1] - thumb.height) // 2
    canvas.paste(thumb, (x, y))
    return canvas


def write_contact_sheet(report: dict[str, object], output: Path, max_candidates: int = 8) -> None:
    candidates = list(report["candidates"])[:max_candidates]
    rows = 1 + len(candidates)
    cell_w, cell_h = 320, 260
    label_h = 28
    sheet = Image.new("RGB", (cell_w * 2, rows * (cell_h + label_h)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    def paste_pair(row: int, left_path: Path, right_path: Path, label: str) -> None:
        y = row * (cell_h + label_h)
        draw.text((8, y + 6), label, fill=(0, 0, 0))
        sheet.paste(_thumbnail(left_path, (cell_w, cell_h)), (0, y + label_h))
        sheet.paste(_thumbnail(right_path, (cell_w, cell_h)), (cell_w, y + label_h))

    real_sample = report["real_samples"][0]
    paste_pair(
        0,
        Path(str(real_sample["paths"]["left_wrist"])),
        Path(str(real_sample["paths"]["right_wrist"])),
        f"real reference {real_sample['label']}: left / right",
    )
    for row, candidate in enumerate(candidates, start=1):
        frame_dir = Path(str(candidate["frame_dir"]))
        score = float(candidate["mean_score"])
        nearest = str(candidate.get("nearest_real_label", "real"))
        paste_pair(
            row,
            _find_role_image(frame_dir, "left_wrist") or frame_dir,
            _find_role_image(frame_dir, "right_wrist") or frame_dir,
            f"sim nearest={score:.4f} ({nearest}): {frame_dir.name}",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", required=True, action="append", type=Path)
    parser.add_argument("--sim-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument("--max-contact-candidates", type=int, default=8)
    parser.add_argument(
        "--rank-by",
        choices=("nearest", "mean-reference"),
        default="nearest",
        help="Sort candidates by nearest real sample or by the mean score across all real samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args.real_dir, args.sim_root, rank_by=args.rank_by)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.contact_sheet is not None:
        write_contact_sheet(report, args.contact_sheet, max_candidates=args.max_contact_candidates)
    best = report.get("best_frame_dir")
    best_score = None
    if report["candidates"]:
        best_score = float(report["candidates"][0]["mean_score"])
    score_text = "nan" if best_score is None or math.isnan(best_score) else f"{best_score:.4f}"
    print(f"best={best} mean_score={score_text}")


if __name__ == "__main__":
    main()
