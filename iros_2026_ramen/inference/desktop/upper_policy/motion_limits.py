"""Reviewed physical-G1 motion envelopes shared by policy launchers.

These values are command-target limits, not Unitree motor ratings.  The
flip-table values were selected from the immutable 30 Hz training dataset
``Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2`` at revision
``0dc47877dfb2efbea796a059c81290c649bc773c``.  They remain well below the
20--30 rad/s target slew used by the pinned official ``xr_teleoperate`` arm
controller while avoiding the material trajectory distortion caused by the
older 1 rad/s and 4 rad/s^2 envelope.
"""

from __future__ import annotations


FLIP_TABLE_ARM_VELOCITY_RAD_S = 1.5
FLIP_TABLE_ARM_ACCELERATION_RAD_S2 = 10.0
FLIP_TABLE_HAND_VELOCITY_FRACTION_S = 1.5
FLIP_TABLE_HAND_ACCELERATION_FRACTION_S2 = 20.0
