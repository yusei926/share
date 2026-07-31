"""Narrow compatibility shims for the pinned Unitree XR runtime.

Unitree modules and released ``logging_mp`` packages have used both the
stdlib-style camel-case and snake-case spellings. Keep this adaptation outside
the pinned checkout and install it before any XR module imports ``logging_mp``.
"""

from __future__ import annotations

from typing import Any


def install_logging_mp_compat() -> None:
    """Provide the v1.5 logging-mp names without replacing its implementation."""

    import logging_mp

    if not hasattr(logging_mp, "get_logger"):
        get_logger = getattr(logging_mp, "getLogger", None)
        if not callable(get_logger):
            raise RuntimeError("logging_mp lacks both get_logger and getLogger")

        def get_logger_compat(name: str | None = None, *, level: int | None = None) -> Any:
            logger = get_logger(name)
            if level is not None:
                logger.setLevel(level)
            return logger

        logging_mp.get_logger = get_logger_compat

    if not hasattr(logging_mp, "getLogger"):
        get_logger = getattr(logging_mp, "get_logger", None)
        if not callable(get_logger):
            raise RuntimeError("logging_mp lacks both get_logger and getLogger")

        def get_logger_camel_compat(name: str | None = None) -> Any:
            return get_logger(name)

        logging_mp.getLogger = get_logger_camel_compat

    if not hasattr(logging_mp, "basic_config"):
        basic_config = getattr(logging_mp, "basicConfig", None)
        if not callable(basic_config):
            raise RuntimeError("logging_mp lacks both basic_config and basicConfig")
        logging_mp.basic_config = basic_config

    if not hasattr(logging_mp, "basicConfig"):
        basic_config = getattr(logging_mp, "basic_config", None)
        if not callable(basic_config):
            raise RuntimeError("logging_mp lacks both basic_config and basicConfig")
        logging_mp.basicConfig = basic_config
