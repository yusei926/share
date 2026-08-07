"""実 G1 で Orchestrator を回す実行 entry point (runtime env required)。

# 前提

- **Runtime env**: Python 3.10 + `unitree_sdk2py` + `cyclonedds` (rclpy は使わない、
  Issue #58 参照)
- **G1 が walking FSM state (立ち姿勢) に既に入っている** こと。Damp からの startup は
  operator が Unitree 標準リモコン等で事前に済ませておく (scope 外)。
- **ROS2 camera driver** が `--topic` に指定した CompressedImage を publish 中
- ハーネス / E-stop / clearance 確保 (安全確認)

# 起動 flow (init → run → shutdown)

    init:
        skill_config.yaml load →
        G1SDKWalkActuator init (= SDK ChannelFactory init) → YoloObbPerception load →
        DetectionStream init → Skill registry (initial=SetupSkill、
            move_to_table=MoveToTable、move_table_base=SampleVLASkill、他=MockSkill) →
        JointStateSource subscribe (Issue #75) →
        G1ArmActuator init + start (Issue #75) →
        Ros2FrameSource subscribe (stereo_view="left"、YOLO 学習と一致) →
        Orchestrator init (initial_skill="setup"、log_sink 開く、
            actuator_send_fn=arm.send_action)
    run:
        orch.run_live(source, hz=30) を main thread で loop、以下を繰り返し:
            source.get() → tick → YOLO → cleaner → state → fire → dispatch
        初回frame待機 / frame freshness を監視。setup skill の Type B
        step() が per-tick に arm 姿勢を送り、SetupSkill.max_dwell_sec
        (= stage dwell 合計) 経過で auto-transition → move_to_table (walk) →
        max_dwell_sec 経過で auto-transition → move_table_base (VLA / mock)。
    shutdown (Ctrl+C / SIGINT / LiveSourceSafetyError):
        arm_actuator.stop() (publish loop 停止) →
        actuator.set_velocity(0,0,0,duration=1) でwalking FSMを維持 →
        DDS camera reader close → log 閉じ
        (姿勢/FSM命令は送らず、Dampはoperatorが明示的に実施 = scope外)

# Usage

    pixi run -e runtime python -m inference.desktop.entrypoint \\
        --interface eth0 \\
        --weight model/yolo_obb/runs/m_lowaug_v3/weights/best.pt \\
        --topic /head/camera/color/image_raw/compressed \\
        --log outputs/orch_run.jsonl

    Skill 姿勢定義 (vx / max_dwell_sec / default_pose_rad 等) は
    `inference/desktop/lower_policy/configs/skill_config.yaml` から読み込む。
    別の YAML を指定したい時は `--skill-config <path>` で override。

# 現状の Skill 実装状況

    setup:             SetupSkill     (Type B、YAML 定義の stage を per-tick で流す
                       腕 pre-motion。Issue #81 で追加、initial skill)
    move_to_table:     MoveToTable    (実 SDK 経由の walk、Type A)
    move_table_base:   SampleVLASkill (Type B、両腕水平 fixed pose を返す。
                       real VLA 実装時に step() を model.predict(obs) に差し替え)
    pick_table_leg / insert_table_leg / rotate_leg_to_tighten / flip_table:
                       MockSkill      (no-op、log のみ)

VLA / GR00T 系の実 model 差し込みは別 Epic。SampleVLASkill は drop-in template。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional, TextIO

import yaml

HEAD_CAMERA_STEREO_VIEW = "left"

# skill_config.yaml の default path (Issue #81 Phase 2b、review comment #82 fix)。
# CWD 相対だと pixi task を repo root 外から呼んだ時に silent に "not found" になるので、
# entrypoint module (この __file__) 相対で resolve する。
_DEFAULT_SKILL_CONFIG: Path = (
    Path(__file__).resolve().parent / "lower_policy" / "configs" / "skill_config.yaml"
)

# ---- 実行時にしか必要ない heavy import は main() 内で lazy に ----
# (SDK / cyclonedds は runtime env にしか無く、module top で import すると
#  test collection や `python -m inference.desktop.entrypoint --help` すら
#  ImportError になる。lazy import で `--help` は main env でも動くように)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"value must be a positive finite number, got {value!r}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--interface",
        type=str,
        default="eth0",
        help="DDS network interface (ip a で確認、実機は eth0/enp2s0 等、sim は lo)",
    )
    p.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="YOLO-OBB weight (.pt) path",
    )
    p.add_argument(
        "--topic",
        type=str,
        default="/head/camera/color/image_raw/compressed",
        help="ROS2 CompressedImage topic 名 (Orin 側の camera driver が publish するもの)",
    )
    p.add_argument(
        "--skill-config",
        type=Path,
        default=_DEFAULT_SKILL_CONFIG,
        help=(
            "Skill 姿勢定義 YAML path (Issue #81 Phase 2b)。setup skill の stage 定義や"
            " move_to_table の vx / max_dwell_sec、move_table_base の default_pose_rad 等の"
            " skill config を集約。数値変更は YAML 編集で完結し、code / CLI arg を触らない。"
            " default は entrypoint module 相対で resolve (CWD 依存無し)"
        ),
    )
    p.add_argument(
        "--hz",
        type=_positive_float,
        default=30.0,
        help="Orchestrator tick rate [Hz] (default 30 = camera fps に合わせる)",
    )
    p.add_argument(
        "--camera-startup-timeout",
        type=_positive_float,
        default=10.0,
        help="最初のcamera frameを待つ上限秒数 (default 10)",
    )
    p.add_argument(
        "--frame-timeout",
        type=_positive_float,
        default=1.0,
        help="歩行中にcamera timestamp更新を待つ上限秒数 (default 1)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="JSONL log path (省略 = log 出さない)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="YOLO device (cuda / cpu / None は auto)",
    )
    p.add_argument(
        "--joint-state-topic",
        type=str,
        default="/joint_states",
        help=(
            "JointState topic 名 (Issue #65 real_hw_bridge_node publish)。"
            " obs['joint_state'] に latest snapshot を乗せる"
        ),
    )
    p.add_argument(
        "--wrist-left-topic",
        type=str,
        default="/wrist_left/camera/color/image_raw/compressed",
        help=(
            "Wrist left camera topic 名 (Issue #75 Orin bringup enable_wrist_cameras=true 時)。"
            " obs['wrist_left_rgb'] に latest snapshot を乗せる (real VLA drop-in で使う想定)"
        ),
    )
    p.add_argument(
        "--wrist-right-topic",
        type=str,
        default="/wrist_right/camera/color/image_raw/compressed",
        help="Wrist right camera topic 名 (同上、obs['wrist_right_rgb'])",
    )
    p.add_argument(
        "--no-wrist-cameras",
        action="store_true",
        help=(
            "Wrist camera source を無効化 (Orin で enable_wrist_cameras=false or"
            " camera-only smoke で使う)。obs に wrist_{left,right}_rgb field 追加されない"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- lazy import (runtime env 以外では ImportError にせず --help を通す) ----
    from inference.desktop.lower_policy.actuators.g1_arm_sdk import G1ArmActuator
    from inference.desktop.lower_policy.actuators.g1_sdk import G1SDKWalkActuator
    from inference.desktop.lower_policy.dispatcher import SkillDispatchLowerPolicy
    from inference.desktop.lower_policy.skills.mock import MockSkill
    from inference.desktop.lower_policy.skills.move_to_table import MoveToTable
    from inference.desktop.lower_policy.skills.sample_vla_skill import SampleVLASkill
    from inference.desktop.lower_policy.skills.setup_skill import SetupSkill
    from inference.desktop.orchestrator import LiveSourceSafetyError, Orchestrator
    from inference.desktop.perception.cleaner import load_cleanup_config
    from inference.desktop.perception.frame_source import Ros2FrameSource
    from inference.desktop.perception.joint_state_source import JointStateSource
    from inference.desktop.perception.stream import DetectionStream
    from inference.desktop.perception.yolo_obb import YoloObbPerception

    if not args.weight.exists():
        sys.exit(f"weight not found: {args.weight}")
    if not args.skill_config.exists():
        sys.exit(f"skill config not found: {args.skill_config}")

    print(f"[init] interface={args.interface} weight={args.weight}", file=sys.stderr)
    print(
        f"[init] topic={args.topic} head_view={HEAD_CAMERA_STEREO_VIEW} "
        f"hz={args.hz} skill_config={args.skill_config}",
        file=sys.stderr,
    )

    # 0) Skill config YAML 読み込み (Issue #81 Phase 2b)。各 skill の from_config
    #    にセクション dict を渡すため entrypoint で 1 回だけ safe_load する。
    skill_cfg_raw = yaml.safe_load(
        args.skill_config.read_text(encoding="utf-8")
    )
    if not isinstance(skill_cfg_raw, dict) or "skills" not in skill_cfg_raw:
        sys.exit(f"skill config invalid: 'skills' section missing in {args.skill_config}")
    skills_section: dict = skill_cfg_raw["skills"]

    # 1) 実 G1 Actuator (walking FSM に既に入ってる前提)。
    #    G1SDKWalkActuator.__init__ 内で ChannelFactoryInitialize(0, interface) が
    #    走り、SDK singleton の DomainParticipant が確保される。以降 Ros2FrameSource
    #    もこの singleton を流用して subscribe する。
    actuator = G1SDKWalkActuator(interface=args.interface)

    # 2) Perception layer
    perception = YoloObbPerception(args.weight, device=args.device)
    cleaner = DetectionStream(load_cleanup_config())

    # 3) Skill registry。全 skill の config は skill_config.yaml から読む (Phase 2b)。
    #    - setup:            Type B、SetupSkill (Issue #81、initial skill、腕 pre-motion)
    #    - move_to_table:    Type A、SDK walk (vx / max_dwell_sec は YAML)
    #    - move_table_base:  Type B、SampleVLASkill (default_pose_rad は YAML)。real VLA
    #      drop-in で SampleVLASkill を Gr00tSkill(model) 等に差し替える想定。
    #    - 他 4 skill:       MockSkill (VLA 未実装、log のみ)
    for required in ("setup", "move_to_table", "move_table_base"):
        if required not in skills_section:
            sys.exit(f"skill config invalid: 'skills.{required}' missing")
    dispatcher = SkillDispatchLowerPolicy(
        {
            "setup": SetupSkill.from_config(skills_section["setup"]),
            "move_to_table": MoveToTable.from_config(
                skills_section["move_to_table"], actuator
            ),
            "move_table_base": SampleVLASkill.from_config(
                skills_section["move_table_base"], skill_name="move_table_base"
            ),
            "pick_table_leg": MockSkill("pick_table_leg"),
            "insert_table_leg": MockSkill("insert_table_leg"),
            "rotate_leg_to_tighten": MockSkill("rotate_leg_to_tighten"),
            "flip_table": MockSkill("flip_table"),
        }
    )

    # 4) JointStateSource (Issue #75)。Orin real_hw_bridge_node が /joint_states を
    #    publish (Issue #65) しているのを cyclonedds direct で subscribe →
    #    orchestrator obs["joint_state"] に latest snapshot が乗る。
    joint_state_source = JointStateSource(topic=args.joint_state_topic)
    print(
        f"[init] joint_state_topic={args.joint_state_topic}", file=sys.stderr
    )

    # 5) G1ArmActuator (Issue #75)。SDK ChannelFactory singleton (actuator init で
    #    確保済) を流用して、公式 motion-mode と同じ rt/arm_sdk を 250Hz publish。
    #    SetupSkill / SampleVLASkill.step が返す action tensor は
    #    orchestrator.actuator_send_fn 経由で arm_actuator.send_action へ流れる。
    arm_actuator = G1ArmActuator()
    arm_actuator.start()
    actuator_send_fn = arm_actuator.send_action
    print("[init] arm actuator started (250Hz rt/arm_sdk publish)", file=sys.stderr)

    # 6) Log sink (JSONL append)
    log_sink: Optional[TextIO] = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_sink = args.log.open("w")

    source: Optional[object] = None
    try:
        # 7) Frame source (ROS2 CompressedImage を cyclonedds direct で subscribe)。
        #    Ros2FrameSource の init 内で SDK ChannelFactory 経由で listener を register。
        #    cyclonedds 側が独自 thread で callback を drain するので、rclpy spin 相当の
        #    外部 thread は不要。topology β 前提 (Desktop 頭脳 / Orin publish)。
        # YOLO weight は LeRobot cam_0 = packed head stereo の左眼 640x480 で学習。
        # 実機でも右眼 / packed 全幅へ切り替えず、同じ左眼だけを入力する。
        source = Ros2FrameSource(
            args.topic,
            stereo_view=HEAD_CAMERA_STEREO_VIEW,
        )

        # 7b) Wrist camera sources (Issue #75、`--no-wrist-cameras` で disable 可能)。
        #     D405 の color imager 出力は 848x480 単一 sensor なので stereo split 不要
        #     (default "packed" = 全幅そのまま)。SampleVLASkill (Mock) は obs 使わないが、
        #     real VLA drop-in で obs["wrist_{left,right}_rgb"] を消費する pipeline を
        #     事前に通しておく (obs pass-through verify)。
        wrist_left_source: Optional[Ros2FrameSource] = None
        wrist_right_source: Optional[Ros2FrameSource] = None
        if not args.no_wrist_cameras:
            wrist_left_source = Ros2FrameSource(args.wrist_left_topic)
            wrist_right_source = Ros2FrameSource(args.wrist_right_topic)
            print(
                f"[init] wrist_left={args.wrist_left_topic}, "
                f"wrist_right={args.wrist_right_topic}",
                file=sys.stderr,
            )

        # 8) Orchestrator (Issue #64 auto-transition, Issue #75 arm/joint_state/wrist、
        #    Issue #81 Phase 3: setup skill が initial、SetupSkill.max_dwell_sec 経過で
        #    move_to_table へ chain auto-transition)
        orch = Orchestrator(
            perception, cleaner, dispatcher,
            initial_skill="setup",
            joint_state_source=joint_state_source,
            wrist_left_source=wrist_left_source,
            wrist_right_source=wrist_right_source,
            actuator_send_fn=actuator_send_fn,
            log_sink=log_sink,
        )

        # Default SIGINT handling raises KeyboardInterrupt.  Catching it below and
        # using this single finally block avoids duplicate stop commands.
        print("[run] starting orchestrator tick loop", file=sys.stderr)
        orch.run_live(
            source,
            hz=args.hz,
            startup_timeout=args.camera_startup_timeout,
            frame_timeout=args.frame_timeout,
        )
    except LiveSourceSafetyError as e:
        print(f"[safety-stop] {e}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        pass
    finally:
        _safe_shutdown(
            actuator, log_sink,
            source=source,
            arm_actuator=arm_actuator,
            extra_sources=[wrist_left_source, wrist_right_source] if not args.no_wrist_cameras else [],
        )


def _safe_shutdown(
    actuator: object,
    log_sink: Optional[TextIO],
    *,
    source: Optional[object] = None,
    arm_actuator: Optional[object] = None,
    extra_sources: Optional[list] = None,
) -> None:
    """arm publish 停止 → walk 停止 → camera reader close → log 閉じる。冪等。

    順序: **arm を先に stop** (LowCmd_ publish 止める) → walk 停止指令 → source close
    → log close。arm を先に止めないと、walk 停止後も arm publish が続いて firmware
    timeout で unpower される (脱力リスク)。

    damp() は呼ばない: 常立ち前提 (Issue #64) では motor unload = 転倒リスク。立ち姿勢の
    release (Damp) は operator が Unitree リモコン等で明示的に実施する (scope 外)。
    HighStand も通常停止には不要なので送らない。

    Camera reader は process 終了へ任せず明示 close する。CycloneDDS listener の
    native receive thread を interpreter teardown 前に停止するため。JointStateSource
    は SDK ChannelFactory singleton 経由なので process 終了時にまとめて解放される。
    """
    if arm_actuator is not None:
        try:
            arm_actuator.stop()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] arm actuator stop failed: {e}", file=sys.stderr)
    try:
        # Damp (FSM 1) is intentionally forbidden here: a standing G1 can
        # collapse when damping is entered.  Zero base velocity keeps the
        # internal walking balancer active and matches MoveToTable.stop().
        actuator.set_velocity(  # type: ignore[attr-defined]
            0.0, 0.0, 0.0, duration=1.0
        )
    except Exception as e:
        print(f"[shutdown] actuator stop failed: {e}", file=sys.stderr)
    if source is not None:
        try:
            source.close()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] camera reader close failed: {e}", file=sys.stderr)
    for extra in (extra_sources or []):
        if extra is None:
            continue
        try:
            extra.close()  # type: ignore[attr-defined]
        except Exception as e:
            print(f"[shutdown] extra source close failed: {e}", file=sys.stderr)
    if log_sink is not None:
        try:
            log_sink.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
