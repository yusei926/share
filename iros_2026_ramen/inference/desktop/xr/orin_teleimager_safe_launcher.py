#!/usr/bin/env python3
"""Start Unitree TeleImager without disrupting already-enumerated cameras.

This is an external compatibility launcher for the upstream ``teleimager``
package.  It intentionally does not modify the upstream checkout.  On the
G1 Orin, upstream ``CameraFinder`` reloads ``uvcvideo`` on every startup;
that operation can temporarily remove RealSense D405 devices.  This launcher
uses the already-enumerated V4L and RealSense devices instead.  The two D405
pipelines are read concurrently: serializing two 30 Hz ``wait_for_frames``
calls can starve one camera until ImageServer's five-second ready timeout.

It controls cameras only.  It never creates Unitree DDS publishers or sends a
robot command.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

import yaml


DEFAULT_CONFIG = Path("/home/unitree/teleimager/cam_config_server.yaml")
DEFAULT_RECORDING_ROOT = Path("/home/unitree/teleimager/lossless_recordings")


def _run_text(command: list[str], *, timeout: float = 10.0) -> str:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout


def v4l_device_serial(node: Path) -> str | None:
    properties = _run_text(
        ["udevadm", "info", "--query=property", "--name", str(node)]
    )
    values = {}
    for line in properties.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values.get("ID_SERIAL_SHORT")


def find_stereo_head_node(nodes: Iterable[Path] | None = None) -> Path | None:
    """Return an optional non-RealSense V4L node exposing 1280x480.

    libuvc may leave its kernel V4L interfaces detached after process exit.
    The node is therefore diagnostic only; the authoritative selection is the
    UVC serial and native MJPEG mode checked by
    :func:`find_stereo_head_uvc_serial`.
    """
    candidates: list[Path] = []
    for node in nodes or sorted(Path("/dev").glob("video*")):
        if not node.name[5:].isdigit():
            continue
        description = _run_text(["v4l2-ctl", "-d", str(node), "-D"])
        formats = _run_text(["v4l2-ctl", "-d", str(node), "--list-formats-ext"])
        if "realsense" in description.lower():
            continue
        if "1280/480" in formats or ("1280x480" in formats):
            candidates.append(node)
    if len(candidates) > 1:
        serials = {v4l_device_serial(node) for node in candidates}
        serials.discard(None)
        if len(serials) == 1:
            return candidates[0]
        rendered = ", ".join(map(str, candidates))
        raise RuntimeError(
            "expected at most one non-RealSense 1280x480 head-camera node; "
            f"found {rendered}"
        )
    return candidates[0] if candidates else None


def unconfigured_realsense_usb_devices() -> list[str]:
    """Describe D405 USB devices whose configuration never completed."""

    result: list[str] = []
    for device in sorted(Path("/sys/bus/usb/devices").glob("*")):
        try:
            vendor = (device / "idVendor").read_text(encoding="ascii").strip()
            product = (device / "idProduct").read_text(encoding="ascii").strip()
        except OSError:
            continue
        if (vendor, product) != ("8086", "0b5b"):
            continue
        try:
            configuration = (
                device / "bConfigurationValue"
            ).read_text(encoding="ascii").strip()
        except OSError:
            configuration = ""
        if configuration:
            continue
        try:
            serial = (device / "serial").read_text(encoding="ascii").strip()
        except OSError:
            serial = "unknown"
        result.append(
            f"usb_path={device.name},usb_descriptor_serial={serial},"
            "bConfigurationValue=unset"
        )
    return result


def inspect_realsense_devices() -> tuple[set[str], list[str]]:
    """Enumerate each RealSense independently so one bad USB slot is visible."""

    import pyrealsense2 as rs

    context = rs.context()
    devices = context.query_devices()
    serials: set[str] = set()
    errors: list[str] = []
    for index in range(len(devices)):
        try:
            device = devices[index]
            serials.add(
                str(device.get_info(rs.camera_info.serial_number))
            )
        except Exception as exc:
            errors.append(
                f"slot={index},error={type(exc).__name__}: {exc}"
            )
    errors.extend(unconfigured_realsense_usb_devices())
    return serials, errors


def available_realsense_serials() -> set[str]:
    serials, _errors = inspect_realsense_devices()
    return serials


def wait_for_realsense(expected: set[str], timeout_s: float) -> set[str]:
    deadline = time.monotonic() + timeout_s
    observed: set[str] = set()
    last_errors: list[str] = []
    while time.monotonic() < deadline:
        try:
            observed, last_errors = inspect_realsense_devices()
        except Exception as exc:
            # A broken USB link can make librealsense fail before it can name
            # a device (for example "failed to set power state"). Preserve
            # that distinction in the service log instead of reporting an
            # unexplained top-level exception.
            last_errors = [f"{type(exc).__name__}: {exc}"]
        if expected.issubset(observed):
            return observed
        # A malformed USB device is not repaired by repeatedly asking
        # librealsense to claim it in the same process. Exit immediately so
        # systemd can create a clean process after its configured delay.
        if last_errors:
            break
        time.sleep(0.5)
    missing = ", ".join(sorted(expected - observed))
    if last_errors:
        raise RuntimeError(
            "D405 devices unavailable "
            f"(expected={sorted(expected)}, observed={sorted(observed)}, "
            f"missing={sorted(expected - observed)}, "
            f"errors={last_errors})"
        )
    raise RuntimeError(f"D405 devices not ready after {timeout_s:.0f}s: missing {missing}")


def find_stereo_head_uvc_serial(
    head_node: Path | None = None,
    serial_override: str | None = None,
) -> str:
    """Correlate the 1280x480 V4L node with its non-RealSense UVC identity.

    Upstream ``UVCCamera`` must be the first and only owner to open the
    device.  Opening a temporary ``uvc.Capture`` here can detach the kernel
    interface and make the immediately following official open fail.
    """

    import uvc

    candidates: list[str] = []
    for device in uvc.device_list():
        serial = str(device.get("serialNumber", ""))
        if not serial or not device.get("uid"):
            continue
        if "realsense" in str(device.get("name", "")).lower():
            continue
        candidates.append(serial)
    node_serial = v4l_device_serial(head_node) if head_node is not None else None
    if serial_override:
        if serial_override not in candidates:
            raise RuntimeError(
                f"configured head UVC serial {serial_override!r} is unavailable; "
                f"observed={candidates or ['none']}"
            )
        if node_serial is not None and serial_override != node_serial:
            raise RuntimeError(
                f"configured head UVC serial {serial_override!r} does not match "
                f"the 1280x480 V4L device serial {node_serial!r}"
            )
        return serial_override

    if node_serial in candidates:
        return str(node_serial)
    if len(candidates) != 1:
        rendered = ", ".join(candidates) or "none"
        raise RuntimeError(
            "could not uniquely correlate the 1280x480 V4L head camera with "
            f"a UVC serial; found {rendered}. Set --head-serial explicitly."
        )
    return candidates[0]


def prepare_config(
    config_path: Path,
    *,
    head_serial_override: str | None = None,
    left_wrist_serial_override: str | None = None,
    right_wrist_serial_override: str | None = None,
) -> tuple[dict, str, set[str]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"invalid TeleImager config: {config_path}")

    head = config.get("head_camera")
    left = config.get("left_wrist_camera")
    right = config.get("right_wrist_camera")
    if not all(isinstance(item, dict) for item in (head, left, right)):
        raise RuntimeError("config must define head_camera, left_wrist_camera, and right_wrist_camera")

    wrist_overrides = (
        left_wrist_serial_override,
        right_wrist_serial_override,
    )
    if any(wrist_overrides) and not all(wrist_overrides):
        raise RuntimeError(
            "left and right wrist serial overrides must be provided together"
        )
    if all(wrist_overrides):
        if left_wrist_serial_override == right_wrist_serial_override:
            raise RuntimeError("left and right wrist serials must be distinct")
        left["serial_number"] = str(left_wrist_serial_override)
        right["serial_number"] = str(right_wrist_serial_override)

    serials = {str(left.get("serial_number", "")), str(right.get("serial_number", ""))}
    serials.discard("")
    if len(serials) != 2:
        raise RuntimeError("two distinct D405 serial_number values are required")

    head_node = find_stereo_head_node()
    configured_head_serial = str(head.get("serial_number", "")).strip() or None
    head_serial = find_stereo_head_uvc_serial(
        head_node,
        serial_override=head_serial_override or configured_head_serial,
    )
    # OpenCV decodes and then re-encodes every 1280x480 MJPEG frame on Orin,
    # which was measured at only 15 unique Hz even though it advertised 30 Hz.
    # libuvc forwards the device's native JPEG while decoding the same frame
    # only for WebRTC, sustaining the physical 30 Hz stream.
    head["type"] = "uvc"
    head["video_id"] = None
    head["physical_path"] = None
    head["serial_number"] = head_serial
    head["enable_zmq"] = True
    head["enable_webrtc"] = True
    left["enable_zmq"] = True
    right["enable_zmq"] = True

    head_identity = str(head_node) if head_node is not None else f"uvc:{head_serial}"
    return config, head_identity, serials


def write_config_atomically(config_path: Path, config: dict) -> None:
    temporary = config_path.with_suffix(config_path.suffix + ".safe-launch.tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    temporary.replace(config_path)


def run(
    config_path: Path,
    d405_timeout_s: float,
    *,
    check_only: bool = False,
    head_serial_override: str | None = None,
    left_wrist_serial_override: str | None = None,
    right_wrist_serial_override: str | None = None,
    recording_root: Path = DEFAULT_RECORDING_ROOT,
    recorder_port: int = 60010,
) -> None:
    # The source checkout is installed in the Orin teleimager environment.
    from teleimager import image_server
    from data.flip_table_data_augmentation.teleop.real.lossless_camera import (
        CameraFrameEnvelope,
        LosslessCameraRecorder,
        RecorderControlServer,
    )

    config, head_node, expected_serials = prepare_config(
        config_path,
        head_serial_override=head_serial_override,
        left_wrist_serial_override=left_wrist_serial_override,
        right_wrist_serial_override=right_wrist_serial_override,
    )
    observed = wait_for_realsense(expected_serials, d405_timeout_s)
    write_config_atomically(config_path, config)

    print(f"[safe-teleimager] head={head_node} d405={','.join(sorted(observed))}", flush=True)
    print("[safe-teleimager] preserving uvcvideo; upstream driver reload is disabled", flush=True)
    if check_only:
        print(
            "[safe-teleimager] camera preflight passed; server was not started",
            flush=True,
        )
        return

    class ExistingDevicesFinder:
        """Minimal CameraFinder interface used by this explicit configuration."""

        def __init__(self, realsense_enable: bool = False, verbose: bool = False):
            del realsense_enable, verbose
            import uvc

            self.video_paths = [str(path) for path in sorted(Path("/dev").glob("video*")) if path.name[5:].isdigit()]
            self.rs_serial_numbers = sorted(available_realsense_serials())
            self.uvc_devices = list(uvc.device_list())

        def is_vpath_exist(self, vpath: str | None) -> bool:
            return vpath is not None and vpath in self.video_paths

        def is_rs_serial_exist(self, serial_number: str | None) -> bool:
            return serial_number is not None and str(serial_number) in self.rs_serial_numbers

        def get_uid_by_sn(self, serial_number: str | None):  # type: ignore[no-untyped-def]
            matches = [
                device.get("uid")
                for device in self.uvc_devices
                if str(device.get("serialNumber", "")) == str(serial_number)
            ]
            if len(matches) > 1:
                raise RuntimeError(f"multiple UVC devices use serial {serial_number}")
            return matches[0] if matches else None

    lossless_recorder = LosslessCameraRecorder(recording_root)

    role_names = {
        "head_camera": "head_stereo",
        "left_wrist_camera": "left_wrist",
        "right_wrist_camera": "right_wrist",
    }

    class FreshFrameImageServer(image_server.ImageServer):
        """Publish each acquired camera frame once, without synthetic repeats."""

        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            self._safe_failure_lock = threading.Lock()
            self._safe_failure: tuple[str, str] | None = None
            super().__init__(*args, **kwargs)

        def _record_failure(self, role: str, exc: object) -> None:
            message = f"{type(exc).__name__}: {exc}"
            with self._safe_failure_lock:
                if self._safe_failure is None:
                    self._safe_failure = (str(role), message)
            image_server.logger_mp.error(
                "[safe-teleimager] fatal camera failure "
                f"role={role} error={message}"
            )
            self._stop_event.set()

        def failure(self) -> tuple[str, str] | None:
            with self._safe_failure_lock:
                return self._safe_failure

        def _update_frames(self, cam_topic, camera):  # type: ignore[no-untyped-def]
            interval = 1.0 / camera.get_fps()
            next_frame_time = time.monotonic()
            while not self._stop_event.is_set():
                try:
                    before_index = (
                        camera._zmq_buffer.latest_index
                        if camera.enable_zmq()
                        else -1
                    )
                    camera._update_frame()
                    after_index = (
                        camera._zmq_buffer.latest_index
                        if camera.enable_zmq()
                        else before_index
                    )
                    if after_index != before_index:
                        with camera._safe_frame_condition:
                            camera._safe_frame_sequence += 1
                            camera._safe_last_frame_ns = time.monotonic_ns()
                            sequence = camera._safe_frame_sequence
                            camera._safe_frame_condition.notify_all()
                        physical_role = role_names.get(str(cam_topic))
                        if physical_role is not None:
                            jpeg_bytes = camera.get_jpeg_bytes()
                            metadata = dict(
                                getattr(camera, "_safe_device_frame_metadata", {})
                            )
                            if jpeg_bytes is not None:
                                lossless_recorder.capture(
                                    CameraFrameEnvelope(
                                        role=physical_role,
                                        usb_serial=str(
                                            getattr(
                                                camera,
                                                "_safe_usb_serial",
                                                "unknown",
                                            )
                                        ),
                                        source_sequence=sequence,
                                        orin_capture_monotonic_ns=int(
                                            metadata.get(
                                                "orin_capture_monotonic_ns",
                                                camera._safe_last_frame_ns,
                                            )
                                        ),
                                        device_frame_counter=metadata.get(
                                            "device_frame_counter"
                                        ),
                                        device_timestamp=metadata.get(
                                            "device_timestamp"
                                        ),
                                        timestamp_domain=metadata.get(
                                            "timestamp_domain"
                                        ),
                                        jpeg=bytes(jpeg_bytes),
                                    )
                                )
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    # Match upstream TeleImager: one acquisition failure stops
                    # the server.  In-place librealsense restart can block
                    # forever after a USB disconnect and leave a dead process
                    # occupying the service slot.
                    camera._safe_last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._record_failure(str(cam_topic), exc)
                    break
                next_frame_time += interval
                delay = next_frame_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame_time = time.monotonic()

        def _zmq_pub(self, cam_topic, camera):  # type: ignore[no-untyped-def]
            """Publish each acquired frame once; never synthesize duplicate FPS."""

            last_sequence = -1
            try:
                while not self._stop_event.is_set():
                    with camera._safe_frame_condition:
                        camera._safe_frame_condition.wait_for(
                            lambda: self._stop_event.is_set()
                            or camera._safe_frame_sequence != last_sequence,
                            timeout=0.5,
                        )
                        sequence = camera._safe_frame_sequence
                    if self._stop_event.is_set():
                        break
                    if sequence == last_sequence:
                        continue
                    jpeg_bytes = camera.get_jpeg_bytes()
                    if jpeg_bytes is None:
                        continue
                    self._zmq_publisher_manager.publish(
                        jpeg_bytes, camera.get_zmq_port()
                    )
                    last_sequence = sequence
            except Exception as exc:
                if self._stop_event.is_set():
                    return
                camera._safe_last_error = f"{type(exc).__name__}: {exc}"
                self._record_failure(f"{cam_topic}:zmq", exc)

    # Avoid CameraFinder.__init__, whose first action is an unconditional
    # ``modprobe -r uvcvideo``.  The devices have been checked above instead.
    image_server.CameraFinder = ExistingDevicesFinder
    server = FreshFrameImageServer(config, realsense_enable=True, camera_finder_verbose=False)
    for role, camera in server._cameras.items():
        if camera is None:
            continue
        camera._safe_frame_condition = threading.Condition()
        camera._safe_frame_sequence = 0
        camera._safe_last_frame_ns = 0
        camera._safe_last_error = None
        role_config = config.get(str(role), {})
        camera._safe_usb_serial = (
            str(role_config.get("serial_number", "unknown"))
            if isinstance(role_config, dict)
            else "unknown"
        )
        camera._safe_device_frame_metadata = {}

        # Preserve the official camera implementation while observing the
        # frame object immediately after acquisition.  The proxies delegate
        # every operation other than the timestamp/counter read.
        if hasattr(camera, "pipeline"):
            inner_pipeline = camera.pipeline

            class TimestampingPipeline:
                def __init__(self, inner, owner):  # type: ignore[no-untyped-def]
                    self._inner = inner
                    self._owner = owner

                def __getattr__(self, name):  # type: ignore[no-untyped-def]
                    return getattr(self._inner, name)

                def wait_for_frames(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                    frames = self._inner.wait_for_frames(*args, **kwargs)
                    capture_ns = time.monotonic_ns()
                    frame = frames.get_color_frame()
                    metadata: dict[str, object] = {
                        "orin_capture_monotonic_ns": capture_ns,
                    }
                    if frame:
                        try:
                            metadata["device_frame_counter"] = int(
                                frame.get_frame_number()
                            )
                        except Exception:
                            pass
                        try:
                            metadata["device_timestamp"] = float(
                                frame.get_timestamp()
                            )
                        except Exception:
                            pass
                        try:
                            metadata["timestamp_domain"] = str(
                                frame.get_frame_timestamp_domain()
                            )
                        except Exception:
                            pass
                    self._owner._safe_device_frame_metadata = metadata
                    return frames

            camera.pipeline = TimestampingPipeline(inner_pipeline, camera)
        elif hasattr(camera, "cap"):
            inner_capture = camera.cap

            class TimestampingCapture:
                def __init__(self, inner, owner):  # type: ignore[no-untyped-def]
                    object.__setattr__(self, "_inner", inner)
                    object.__setattr__(self, "_owner", owner)

                def __getattr__(self, name):  # type: ignore[no-untyped-def]
                    return getattr(self._inner, name)

                def __setattr__(self, name, value):  # type: ignore[no-untyped-def]
                    setattr(self._inner, name, value)

                def get_frame_robust(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                    frame = self._inner.get_frame_robust(*args, **kwargs)
                    metadata: dict[str, object] = {
                        "orin_capture_monotonic_ns": time.monotonic_ns(),
                    }
                    if frame is not None:
                        for attribute in ("index", "frame_index"):
                            value = getattr(frame, attribute, None)
                            if value is not None:
                                try:
                                    metadata["device_frame_counter"] = int(value)
                                except (TypeError, ValueError):
                                    pass
                                break
                        value = getattr(frame, "timestamp", None)
                        if value is not None:
                            try:
                                metadata["device_timestamp"] = float(value)
                                metadata["timestamp_domain"] = "uvc"
                            except (TypeError, ValueError):
                                pass
                    self._owner._safe_device_frame_metadata = metadata
                    return frame

            camera.cap = TimestampingCapture(inner_capture, camera)

    def stop(_signum, _frame):  # type: ignore[no-untyped-def]
        print("[safe-teleimager] shutdown requested", flush=True)
        server.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    recorder_control = None
    server.start()
    if server._stop_event.is_set() and server.failure() is None:
        server._record_failure(
            "startup",
            RuntimeError("one or more cameras failed the upstream ready check"),
        )
    if not server._stop_event.is_set():
        recorder_control = RecorderControlServer(
            lossless_recorder,
            port=recorder_port,
        )
        recorder_control.start()
        print(
            "[safe-teleimager] lossless recorder ready "
            f"port={recorder_port} root={recording_root}",
            flush=True,
        )

    def report_camera_health() -> None:
        while not server._stop_event.wait(1.0):
            now_ns = time.monotonic_ns()
            parts: list[str] = []
            for role, camera in server._cameras.items():
                if camera is None:
                    parts.append(f"{role}[unavailable]")
                    continue
                last_ns = int(camera._safe_last_frame_ns)
                age_ms = (
                    None
                    if last_ns <= 0
                    else max(0.0, (now_ns - last_ns) / 1.0e6)
                )
                role_config = config.get(str(role), {})
                serial = (
                    role_config.get("serial_number")
                    if isinstance(role_config, dict)
                    else None
                )
                parts.append(
                    f"{role}[serial={serial or 'none'},"
                    f"sequence={int(camera._safe_frame_sequence)},"
                    f"age_ms={'none' if age_ms is None else f'{age_ms:.1f}'},"
                    f"error={camera._safe_last_error!r}]"
                )
            print(
                "[safe-teleimager] health " + " ".join(parts),
                flush=True,
            )
            recorder_status = lossless_recorder.status()
            if recorder_status["active"] or recorder_status["failed"]:
                print(
                    "[safe-teleimager] recorder "
                    + json.dumps(recorder_status, sort_keys=True),
                    flush=True,
                )

    health_thread = threading.Thread(
        target=report_camera_health,
        name="safe-teleimager-health",
        daemon=True,
    )
    health_thread.start()
    try:
        server.wait()
    finally:
        server.stop()
        health_thread.join(timeout=2.0)
        if recorder_control is not None:
            recorder_control.close()
        lossless_recorder.close()
    failure = server.failure()
    if failure is not None:
        role, error = failure
        raise RuntimeError(
            f"camera server stopped after role failure: role={role} error={error}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--d405-timeout", type=float, default=30.0)
    parser.add_argument(
        "--recording-root",
        type=Path,
        default=Path(
            os.environ.get(
                "IROS_AVP_RECORDING_ROOT",
                str(DEFAULT_RECORDING_ROOT),
            )
        ),
    )
    parser.add_argument(
        "--recorder-port",
        type=int,
        default=int(os.environ.get("IROS_AVP_RECORDER_PORT", "60010")),
    )
    parser.add_argument(
        "--head-serial",
        default=os.environ.get("HEAD_CAMERA_SERIAL"),
        help="explicit UVC serial when multiple non-RealSense cameras exist",
    )
    parser.add_argument(
        "--left-wrist-serial",
        default=os.environ.get("WRIST_LEFT_SERIAL"),
        help="machine-local serial physically mounted on the left wrist",
    )
    parser.add_argument(
        "--right-wrist-serial",
        default=os.environ.get("WRIST_RIGHT_SERIAL"),
        help="machine-local serial physically mounted on the right wrist",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not args.config.is_file():
        raise SystemExit(f"config does not exist: {args.config}")
    try:
        run(
            args.config,
            args.d405_timeout,
            check_only=args.check_only,
            head_serial_override=args.head_serial,
            left_wrist_serial_override=args.left_wrist_serial,
            right_wrist_serial_override=args.right_wrist_serial,
            recording_root=args.recording_root,
            recorder_port=args.recorder_port,
        )
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[safe-teleimager] fatal: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
