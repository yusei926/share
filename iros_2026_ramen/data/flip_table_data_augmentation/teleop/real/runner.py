"""Entrypoint for physical G1 + Dex1 AVP teleoperation."""

from __future__ import annotations

from ..session import main as shared_session_main


def main() -> int:
    return shared_session_main(forced_backend="real")


if __name__ == "__main__":
    raise SystemExit(main())
