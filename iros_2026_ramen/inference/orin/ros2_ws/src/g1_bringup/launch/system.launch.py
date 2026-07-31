import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


DEFAULT_HEAD_CAMERA_DEVICE = (
    "/dev/v4l/by-id/"
    "usb-USB2.0_Camera_RGB_USB2.0_Camera_RGB_01.00.00-video-index0"
)

# Issue #75 実機疎通機体 (2026-07-19 作業員 report) の D405 serial。
# 左右 mapping は operator が実映像で確認後に決定 (現状は物理 bus 番号順の初期 guess:
# bus 2-2.1 = 128422271048 = 左、bus 2-2.2 = 128422271925 = 右)。
# 実映像で swap が必要なら WRIST_LEFT_SERIAL / WRIST_RIGHT_SERIAL 環境変数 or
# launch arg で override 可能。
# RealSense API が報告するD405固有serial。/dev/v4l/by-idの文字列とは異なる。
# 実機で確認: bus 2-2.1 = 128422271048、bus 2-2.2 = 128422271925。
DEFAULT_WRIST_LEFT_SERIAL = "128422271048"
DEFAULT_WRIST_RIGHT_SERIAL = "128422271925"


def _launch_head_camera(context):
    configured_device = LaunchConfiguration("head_camera_device").perform(context)
    # usb_cam 0.6.x mishandles relative /dev/v4l/by-id symlinks, so retain the
    # stable external configuration but hand the driver a canonical /dev/videoN.
    resolved_device = os.path.realpath(configured_device)

    return [
        LogInfo(msg=(
            f"Head camera device: {configured_device} -> {resolved_device}"
        )),
        Node(
            package="usb_cam",
            executable="usb_cam_node_exe",
            namespace="head/camera/color",
            parameters=[{
                "video_device": resolved_device,
                "pixel_format": "mjpeg2rgb",
                "image_width": 1280,
                "image_height": 480,
                "framerate": 30.0,
                "camera_name": "head_camera",
                "camera_frame_id": "head_camera_optical_frame",
                # 運営データ収集コードと圧縮条件を揃える (default 95 → 90)。
                "image_raw.compressed.jpeg_quality": 90,
            }],
            output="screen",
        ),
    ]


def _launch_wrist_camera(context, side: str):
    """side = 'left' or 'right'。D405 は RealSense proprietary protocol なので
    usb_cam ではなく realsense2_camera driver を使う。namespace は
    `/wrist_{side}/camera` で、head camera (`/head/camera/color`) と揃える形。

    CLAUDE.md Key Facts: **D405 IR projector is OFF** (passive stereo design)。
    depth_module.emitter_enabled=0 で projector 無効化 (dataset G1_WBT と整合)。

    Publish topics (namespace prefix `/wrist_{side}/camera/` 付き):
      - color/image_raw + color/image_raw/compressed (mono RGB、YOLO 学習入力形式)
      - infra1/image_raw + infra1/image_raw/compressed (left IR)
      - infra2/image_raw + infra2/image_raw/compressed (right IR、stereo pair)
    depth は enable_depth=false で無効 (compute cost 節約、必要になったら enable)。
    """
    serial = LaunchConfiguration(f"wrist_{side}_serial").perform(context)
    return [
        LogInfo(msg=f"Wrist {side} camera: serial={serial}"),
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            # realsense2_camera自身がcamera階層を持つため、namespaceはwrist側まで。
            # これで /wrist_{side}/camera/color/image_raw になる。
            namespace=f"wrist_{side}",
            parameters=[{
                # realsense2_camera 慣例: serial 前に "_" prefix 付ける
                "serial_no": f"_{serial}",
                "enable_color": True,
                "enable_infra1": True,
                "enable_infra2": True,
                "enable_depth": False,          # depth compute しない (YAGNI)
                "depth_module.emitter_enabled": 0,  # IR projector OFF (CLAUDE.md 確定)
                # D405 の許容範囲は [0, 2]。driver default の 3 はこの機種では
                # reject されるため、無効 (0) を明示して起動 warning を避ける。
                "depth_module.power_line_frequency": 0,
                # D405がこの実機で広告するRGB/IR profile。640幅を要求すると
                # driverが暗黙に848x480へfallbackするため、実値を明示する。
                "rgb_camera.color_profile": "848x480x30",
                "depth_module.depth_profile": "848x480x30",  # IR stereo 用 profile
                "camera_name": f"wrist_{side}_camera",
                "camera_frame_id": f"wrist_{side}_camera_optical_frame",
                # JPEG compression 条件は head camera と揃える
                "color.image_raw.compressed.jpeg_quality": 90,
                "infra1.image_raw.compressed.jpeg_quality": 90,
                "infra2.image_raw.compressed.jpeg_quality": 90,
            }],
            output="screen",
        ),
    ]


def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare("g1_bringup"),
        "config",
        "system.yaml",
    ])

    mock_hardware = LaunchConfiguration("mock_hardware")
    enable_camera = LaunchConfiguration("enable_camera")
    enable_wrist_cameras = LaunchConfiguration("enable_wrist_cameras")
    common_output = "screen"

    return LaunchDescription([
        DeclareLaunchArgument(
            "mock_hardware",
            default_value="true",
            description=(
                "true: safe mock hardware bridge (実機不要、zero joint publish)。"
                " false: real Unitree bridge (unitree_sdk2py 経由で rt/lowstate 購読)。"
            ),
        ),
        DeclareLaunchArgument(
            "enable_camera",
            default_value="true",
            description=(
                "true: head USB camera を起動。false: camera 無しで hardware bridge のみ起動。"
            ),
        ),
        DeclareLaunchArgument(
            "head_camera_device",
            default_value=EnvironmentVariable(
                "HEAD_CAMERA_DEVICE",
                default_value=DEFAULT_HEAD_CAMERA_DEVICE,
            ),
            description=(
                "Head camera capture device. Prefer /dev/v4l/by-id; override "
                "for competition hardware with HEAD_CAMERA_DEVICE or this argument."
            ),
        ),
        # Issue #75 wrist camera (D405 x 2) launch args
        DeclareLaunchArgument(
            "enable_wrist_cameras",
            default_value="false",
            description=(
                "true: 両手 wrist RealSense D405 を起動 (RGB + IR stereo)。"
                " false (default): 起動しない (head camera 単体で pipeline verify したい時)。"
                " Issue #75 実機 verify Step 1 で true に切り替えて疎通確認。"
            ),
        ),
        DeclareLaunchArgument(
            "wrist_left_serial",
            default_value=EnvironmentVariable(
                "WRIST_LEFT_SERIAL", default_value=DEFAULT_WRIST_LEFT_SERIAL
            ),
            description=(
                "Wrist left camera serial (D405)。実映像で左右対応付け確認後、"
                " 間違ってたらこの arg (or WRIST_LEFT_SERIAL env) で swap 可能。"
                " default は Issue #75 疎通機体 (2026-07-19) の bus 2-2.1 側 serial。"
            ),
        ),
        DeclareLaunchArgument(
            "wrist_right_serial",
            default_value=EnvironmentVariable(
                "WRIST_RIGHT_SERIAL", default_value=DEFAULT_WRIST_RIGHT_SERIAL
            ),
            description=(
                "Wrist right camera serial (D405)。default は Issue #75 疎通機体の"
                " bus 2-2.2 側 serial。左右間違ってたら wrist_left_serial と swap。"
            ),
        ),
        # mock bridge (`mock_hardware:=true` の default)。実機無しで /joint_states 等を zero publish。
        Node(
            package="g1_hw_bridge",
            executable="mock_hw_bridge_node",
            name="g1_mock_hw_bridge",
            output=common_output,
            parameters=[config],
            condition=IfCondition(mock_hardware),
        ),
        # real bridge (`mock_hardware:=false`)。G1 body 電源 ON + EtherCAT 接続後に使用。
        # motor 命令は発行しない (subscribe only) ので Damp state で安全に joint 位置を
        # Desktop 側に流せる。walk 動作テストは Issue #64 (Phase 2 #C) で。
        Node(
            package="g1_hw_bridge",
            executable="real_hw_bridge_node",
            name="g1_real_hw_bridge",
            output=common_output,
            parameters=[config],
            condition=UnlessCondition(mock_hardware),
        ),
        # 実 HBVCAM (USB2.0 Camera RGB) を usb_cam で grab (MJPG decode → raw Image)、
        # image_transport-plugins が /image_raw/compressed (CompressedImage) を自動生成する。
        # namespace=head/camera/color を付けて Desktop 側 Ros2FrameSource が subscribe する
        # /head/camera/color/image_raw/compressed に一致する topic 名にする。
        # v4l2_camera (humble binary 0.6.2) は MJPG 未対応で crash するため usb_cam に切替。
        OpaqueFunction(
            function=_launch_head_camera,
            condition=IfCondition(enable_camera),
        ),
        # Wrist camera 2 台 (D405、RealSense driver 経由)。
        # `enable_wrist_cameras:=true` で opt-in。CLAUDE.md 確定事項: IR projector OFF
        # (depth_module.emitter_enabled=0 で dataset G1_WBT の passive stereo と整合)。
        OpaqueFunction(
            function=lambda ctx: _launch_wrist_camera(ctx, "left"),
            condition=IfCondition(enable_wrist_cameras),
        ),
        OpaqueFunction(
            function=lambda ctx: _launch_wrist_camera(ctx, "right"),
            condition=IfCondition(enable_wrist_cameras),
        ),
    ])
