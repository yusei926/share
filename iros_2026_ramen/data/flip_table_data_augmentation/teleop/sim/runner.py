"""Entrypoint for Isaac/RoboFinals AVP teleoperation."""

from __future__ import annotations

from ..session import main as shared_session_main


def main() -> int:
    return shared_session_main(forced_backend="sim")


if __name__ == "__main__":
    raise SystemExit(main())
