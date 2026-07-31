from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path
from typing import Any

from .config import CurationConfig
from .source import RGB_KEYS, download_source


def _build_tile(
    snapshot,
    row: dict[str, Any],
    output: Path,
    *,
    start_frame: int,
    end_frame: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        return
    duration = max(1, end_frame - start_frame) / snapshot.fps
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for key in RGB_KEYS:
        seek = snapshot.video_offset(row, key) + start_frame / snapshot.fps
        command.extend(
            ["-ss", f"{seek:.9f}", "-t", f"{duration:.9f}", "-i", str(snapshot.video_path(row, key))]
        )
    command.extend(
        [
            "-filter_complex",
            "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];"
            "[top][bottom]vstack,fps=30,scale=960:720[out]",
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    subprocess.run(command, check=True)


def generate_review(config: CurationConfig) -> Path:
    analysis_path = config.workspace / "analysis" / "analysis.json"
    if not analysis_path.is_file():
        raise FileNotFoundError("run analyze first")
    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    snapshot = download_source(config, include_videos=True, rgb_only=True)
    rows = {int(row["episode_index"]): row for row in snapshot.episodes}
    records = {
        int(record["source_episode_index"]): record for record in report["records"]
    }
    review_root = config.workspace / "review"
    videos = review_root / "videos"
    cluster_sections: list[str] = []
    for family in ("orientation", "trajectory"):
        section_parts = [f"<h2>{html.escape(family)} clusters</h2>"]
        sizes = report[family]["cluster_sizes"]
        for label, episodes in sorted(
            report[family]["representatives"].items(), key=lambda item: int(item[0])
        ):
            section_parts.append(
                f"<h3>cluster {html.escape(label)} (n={sizes.get(label, '?')})</h3>"
            )
            section_parts.append('<div class="grid">')
            for episode in episodes:
                episode = int(episode)
                record = records[episode]
                output = videos / f"episode_{episode:06d}.mp4"
                _build_tile(
                    snapshot,
                    rows[episode],
                    output,
                    start_frame=int(record["trim_start"]),
                    end_frame=int(record["trim_end"]),
                )
                section_parts.append(
                    "<figure>"
                    f'<video controls preload="metadata" src="videos/{output.name}"></video>'
                    f"<figcaption>episode {episode}; stability="
                    f"{record['trajectory_stability']:.3f}; steps={record['step_count']}</figcaption>"
                    "</figure>"
                )
            section_parts.append("</div>")
        cluster_sections.append("\n".join(section_parts))
    suggested = report["suggested_decision"]
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>flip_table_2 curation review</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; background: #111; color: #eee; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }}
figure {{ margin: 0; padding: .5rem; background: #222; }}
video {{ width: 100%; }}
code {{ color: #9f9; }}
</style>
<h1>flip_table_2 representative review</h1>
<p>suggested orientation=<code>{suggested['orientation_cluster_ids']}</code>,
trajectory=<code>{suggested['trajectory_cluster_id']}</code></p>
{''.join(cluster_sections)}
"""
    output = review_root / "index.html"
    output.write_text(page, encoding="utf-8")
    print(f"[review] open file://{output}")
    return output

