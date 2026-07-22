"""Narrow compatibility shims for the pinned Unitree XR runtime.

The v1.5 XR source calls the historical ``logging_mp`` snake-case API while
the supported 0.2.x package exports the same behavior as camel-case methods.
Keep this adaptation outside the pinned checkout and install it before any
Unitree, TeleVuer, or TeleImager module imports ``logging_mp``.
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

    if not hasattr(logging_mp, "basic_config"):
        basic_config = getattr(logging_mp, "basicConfig", None)
        if not callable(basic_config):
            raise RuntimeError("logging_mp lacks both basic_config and basicConfig")
        logging_mp.basic_config = basic_config
