from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_labels
from .build import build_dataset
from .config import DEFAULT_CONFIG_PATH, load_config
from .publish import publish
from .validate import validate_local


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-labels")
    build = commands.add_parser("build")
    build.add_argument("--max-episodes", type=int)
    validate = commands.add_parser("validate")
    validate.add_argument("--minimum-episodes", type=int)
    commands.add_parser("publish")
    return result


def main() -> int:
    args = parser().parse_args()
    config = load_config(args.config)
    if args.command == "audit-labels":
        value = audit_labels(config)
    elif args.command == "build":
        value = {"root": str(build_dataset(config, maximum_episodes=args.max_episodes))}
    elif args.command == "validate":
        value = validate_local(config, minimum_episodes=args.minimum_episodes)
    elif args.command == "publish":
        value = publish(config)
    else:
        raise AssertionError(args.command)
    print(json.dumps(value, ensure_ascii=False, indent=2)[:10000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
