#!/usr/bin/env python3
"""Generate deterministic, tileable room textures without external assets."""

from __future__ import annotations

import math
from pathlib import Path
import random
import struct
import zlib


SIZE = 256
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "room" / "textures"


def _write_png(path: Path, pixels: list[tuple[int, int, int]]) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = []
    for y in range(SIZE):
        row = bytearray([0])
        for r, g, b in pixels[y * SIZE : (y + 1) * SIZE]:
            row.extend((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
        rows.append(bytes(row))
    header = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def _noise(seed: int) -> random.Random:
    return random.Random(seed)


def wood() -> list[tuple[int, int, int]]:
    rng = _noise(11)
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            grain = 13.0 * math.sin(2.0 * math.pi * (x / 42.0 + 0.08 * math.sin(y / 27.0)))
            fine = 5.0 * math.sin(2.0 * math.pi * x / 7.0)
            seam = -28.0 if y % 64 in (0, 1) else 0.0
            n = rng.uniform(-5.0, 5.0)
            pixels.append((int(146 + grain + fine + seam + n), int(91 + 0.55 * grain + seam + n), int(49 + 0.25 * grain + seam)))
    return pixels


def concrete() -> list[tuple[int, int, int]]:
    rng = _noise(23)
    pixels = []
    for _y in range(SIZE):
        for _x in range(SIZE):
            n = rng.gauss(0.0, 10.0)
            speck = rng.choice((-28.0, 24.0)) if rng.random() < 0.018 else 0.0
            value = int(151 + n + speck)
            pixels.append((value, value + 1, value + 3))
    return pixels


def ceramic_tile() -> list[tuple[int, int, int]]:
    rng = _noise(37)
    pixels = []
    cell = 64
    for y in range(SIZE):
        for x in range(SIZE):
            grout = x % cell < 4 or y % cell < 4
            if grout:
                pixels.append((92, 96, 98))
            else:
                shade = int(rng.uniform(-5, 6) + 5 * math.sin((x + y) / 31.0))
                pixels.append((194 + shade, 202 + shade, 204 + shade))
    return pixels


def brick() -> list[tuple[int, int, int]]:
    rng = _noise(41)
    pixels = []
    brick_w, brick_h, grout = 64, 32, 4
    for y in range(SIZE):
        row = y // brick_h
        shifted_x = (x_offset := (brick_w // 2 if row % 2 else 0))
        for x in range(SIZE):
            local_x = (x + shifted_x) % brick_w
            local_y = y % brick_h
            if local_x < grout or local_y < grout:
                pixels.append((151, 143, 130))
            else:
                n = int(rng.uniform(-11, 12))
                pixels.append((151 + n, 72 + n // 2, 49 + n // 3))
    return pixels


def plaster() -> list[tuple[int, int, int]]:
    rng = _noise(53)
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            n = rng.gauss(0.0, 3.0) + 2.0 * math.sin((x + 2 * y) / 47.0)
            pixels.append((int(218 + n), int(217 + n), int(209 + n)))
    return pixels


def vinyl() -> list[tuple[int, int, int]]:
    rng = _noise(67)
    pixels = []
    for y in range(SIZE):
        for x in range(SIZE):
            seam = -18 if x % 48 in (0, 1) else 0
            n = int(rng.uniform(-4, 5) + 3 * math.sin(y / 19.0))
            pixels.append((116 + n + seam, 128 + n + seam, 127 + n + seam))
    return pixels


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generators = {
        "oak_wood.png": wood,
        "rough_concrete.png": concrete,
        "ceramic_tile.png": ceramic_tile,
        "red_brick.png": brick,
        "painted_plaster.png": plaster,
        "industrial_vinyl.png": vinyl,
    }
    for name, generator in generators.items():
        _write_png(OUTPUT_DIR / name, generator())
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
