"""Process-wide exclusion for physical G1 command publishers."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


G1_CONTROL_LOCK_PATH = Path(
    os.environ.get(
        "IROS_G1_CONTROL_LOCK_PATH",
        f"/run/user/{os.getuid()}/iros_2026_ramen_g1_control.lock",
    )
)
_LOCK_FD: int | None = None


def _validate_lock_fd(fd: int) -> int:
    try:
        fd_stat = os.fstat(fd)
        path_stat = G1_CONTROL_LOCK_PATH.stat()
    except OSError as exc:
        raise RuntimeError("invalid G1 control-lock descriptor") from exc
    if (fd_stat.st_dev, fd_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
        raise RuntimeError("G1 control lock does not match the canonical inode")
    if path_stat.st_uid != os.getuid():
        raise RuntimeError("G1 control lock is not owned by the current user")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("G1 control-lock descriptor does not own the lock") from exc
    return fd


def _validated_inherited_fd() -> int | None:
    raw = os.environ.get("IROS_G1_CONTROL_LOCK_FD", "").strip()
    if not raw:
        return None
    try:
        fd = int(raw)
    except ValueError as exc:
        raise RuntimeError("invalid inherited G1 control-lock descriptor") from exc
    return _validate_lock_fd(fd)


def current_g1_control_lock_fd() -> int:
    """Return the verified descriptor for a same-session child process."""

    if _LOCK_FD is None:
        raise RuntimeError("G1 control lock has not been acquired")
    return _validate_lock_fd(_LOCK_FD)


def acquire_g1_control_lock() -> None:
    """Fail closed when another repository G1 controller owns the robot."""

    global _LOCK_FD
    if _LOCK_FD is not None:
        return
    inherited = _validated_inherited_fd()
    if inherited is not None:
        _LOCK_FD = inherited
        return
    G1_CONTROL_LOCK_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(G1_CONTROL_LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(fd)
        raise RuntimeError(
            "another IROS RAMEN physical-G1 controller is already running "
            f"(lock={G1_CONTROL_LOCK_PATH})"
        ) from exc
    _LOCK_FD = fd
