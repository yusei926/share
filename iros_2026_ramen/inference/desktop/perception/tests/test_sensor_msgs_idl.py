"""sensor_msgs_idl.CompressedImage_ の IDL 定義 test (Issue #58)。

cyclonedds が install されている runtime env でのみ動く。main env では
importorskip で skip する。

test 内容:
- typename が ROS2 canonical (`sensor_msgs::msg::dds_::CompressedImage_`) と一致
- field の順序と名前が ROS2 msg 定義と一致 (header / format / data)
- header が unitree_sdk2py の Header_ を参照している
"""

from __future__ import annotations

from dataclasses import fields

import pytest

cyclonedds = pytest.importorskip("cyclonedds")

from inference.desktop.perception.sensor_msgs_idl import CompressedImage_


class TestCompressedImageIdl:
    def test_typename_matches_ros2_canonical(self):
        """DDS typename が ROS2 canonical と一致することを確認。

        wire 互換の観点で重要: rclpy publisher が流す msg と subscribe できるかは
        typename の完全一致に依存する。
        """
        # cyclonedds IdlStruct では typename は class attr `__idl_typename__` に入る
        assert (
            CompressedImage_.__idl_typename__
            == "sensor_msgs::msg::dds_::CompressedImage_"
        )

    def test_fields_match_ros2_msg_definition(self):
        """field 順序と名前が ROS2 sensor_msgs/CompressedImage.msg と一致。

        ROS2 msg:
            std_msgs/Header header
            string format
            uint8[] data
        """
        field_names = [f.name for f in fields(CompressedImage_)]
        assert field_names == ["header", "format", "data"]

    def test_header_type_is_sdk_header(self):
        """header field type が unitree_sdk2py の Header_ を参照している (typename 一致で流用)。"""
        header_field = next(f for f in fields(CompressedImage_) if f.name == "header")
        # forward reference string で書かれている (import order の都合)
        # 実際の型は unitree_sdk2py.idl.std_msgs.msg.dds_.Header_
        assert "std_msgs" in str(header_field.type)
        assert "Header_" in str(header_field.type)

    def test_idl_runtime_type_resolution(self):
        """CycloneDDS が nested Header_ を実際に解決できることを確認。"""
        # Topic 作成時に呼ばれる処理。型名の引用符など、dataclass の field
        # inspection だけでは検出できない runtime error をここで捕捉する。
        CompressedImage_.__idl__.populate()
