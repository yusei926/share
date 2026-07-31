"""Pinned dataset frame-zero arm/hand poses for physical subtask evaluation.

The pose is the per-joint median of ``action.robot_q_desired[22:36]`` at
``frame_index == 0`` over the training episodes used by the model.  A median
is used because a dataset has no single canonical episode zero and it is
robust to small operator-to-operator differences.  These values command only
the fourteen arm joints; Regular Mode remains the sole owner of waist/legs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class SubtaskStartPose:
    arm_position_rad: tuple[float, ...]
    dataset_repo_id: str
    dataset_revision: str
    training_episode_count: int
    statistic: str = "median_action_robot_q_desired_frame0_arms14"
    exact_training_revision: bool = True
    dex1_opening_fraction: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        arm = np.asarray(self.arm_position_rad, dtype=np.float64)
        if arm.shape != (14,) or not np.isfinite(arm).all():
            raise ValueError("subtask start pose must be finite 14-D")
        if not self.dataset_repo_id or not self.dataset_revision:
            raise ValueError("subtask start pose requires dataset provenance")
        if self.training_episode_count < 1:
            raise ValueError("training episode count must be positive")
        if self.dex1_opening_fraction is not None:
            hand = np.asarray(self.dex1_opening_fraction, dtype=np.float64)
            if (
                hand.shape != (2,)
                or not np.isfinite(hand).all()
                or np.any((hand < 0.0) | (hand > 1.0))
            ):
                raise ValueError(
                    "subtask start Dex1 opening must be finite [2] in [0,1]"
                )

    @property
    def sha256(self) -> str:
        payload = {
            "arm_position_rad": list(self.arm_position_rad),
            "dataset_repo_id": self.dataset_repo_id,
            "dataset_revision": self.dataset_revision,
            "training_episode_count": self.training_episode_count,
            "statistic": self.statistic,
            "exact_training_revision": self.exact_training_revision,
            "dex1_opening_fraction": (
                None
                if self.dex1_opening_fraction is None
                else list(self.dex1_opening_fraction)
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


PICK_LEG_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        -0.3292621821165085, 0.4412737190723419, -0.16212594509124756,
        -0.15982457995414734, -0.16640856117010117, 0.5013763606548309,
        -0.8052002191543579, -0.668845921754837, -0.27757230401039124,
        -0.05611474625766277, 0.41018396615982056, 0.08590780571103096,
        0.7006275355815887, 0.2249610498547554,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1",
    dataset_revision="4a5aef3bc43470b636f5c9d4f9102a30311b79a5",
    training_episode_count=2114,
)

# The ACT deployment source deliberately selected episode 2101 frame 0 as a
# representative late-dataset medoid (spec v2), instead of the all-episode
# median used by the GR00T checkpoints above.  Preserve that exact choice.
PICK_LEG_ACT_EP2101_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        -0.2438, 0.4255, -0.1550, -0.1365, -0.1600, 0.4605, -0.7208,
        -0.6929, -0.2517, -0.0821, 0.4797, 0.0727, 0.7051, 0.1712,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_pick_leg_1",
    dataset_revision="4a5aef3bc43470b636f5c9d4f9102a30311b79a5",
    training_episode_count=1806,
    statistic=(
        "deployment_medoid_episode2101_frame0_action_arms14;"
        "source_commit=817e8addd944b43b9ada6d096feaa93f75179d38"
    ),
)

COARSE_INSERT_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        -0.5638502836227417, 0.4385502338409424, -0.35575076937675476,
        0.39411237835884094, -0.052953992038965225, 0.19772784411907196,
        -1.057064414024353, -0.34810954332351685, -0.2936753034591675,
        0.021565500646829605, -0.011338683776557446, 0.3509312570095062,
        0.49885404109954834, 0.5943645238876343,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_1",
    dataset_revision="338c3db7f106e5108f29df5bb564a983897c0eca",
    training_episode_count=1907,
    # Median frame-zero action.hand_cmd over the exact 1,907 episode indices
    # embedded in this checkpoint's train_config, converted from 0..4.5.
    dex1_opening_fraction=(
        0.29191909896002877,
        0.4211886458926731,
    ),
    statistic=(
        "median_action_robot_q_desired_frame0_arms14;"
        "median_action_hand_cmd_frame0_dex1_2;"
        "episode_indices_from_checkpoint_train_config"
    ),
    # The checkpoint's train_config did not pin the dataset revision. This is
    # the current dataset revision and must not be represented as exact.
    exact_training_revision=False,
)

FLIP_TABLE_V2_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        0.0207520704716444, 0.18185654282569885, 0.266309529542923,
        -0.08172374218702316, -0.1415475308895111, 0.0874384418129921,
        -0.22016701102256775, -0.055108800530433655, -0.2730565667152405,
        0.07833772897720337, -0.07368876785039902, 0.32239586114883423,
        0.26626819372177124, 0.7089221477508545,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2",
    dataset_revision="0dc47877dfb2efbea796a059c81290c649bc773c",
    training_episode_count=171,
    # Median action.hand_cmd at frame 0 over the same 171 training episodes,
    # converted from the dataset's 0..4.5 scalar to physical opening fraction.
    dex1_opening_fraction=(0.926145871480306, 1.0),
    statistic=(
        "median_action_robot_q_desired_frame0_arms14;"
        "median_action_hand_cmd_frame0_dex1_2"
    ),
)

FLIP_TABLE_V1_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        -0.0154762147, 0.1663854271, 0.2027824968, -0.1144016460,
        -0.1372865140, 0.1624660343, -0.2493835539, -0.0566369109,
        -0.2420272082, 0.0551154912, -0.1077821404, 0.2883513272,
        0.2672411799, 0.6056411862,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_1",
    dataset_revision="10a6ec05f9993b8d59faad2957e47153b0f15f37",
    training_episode_count=531,
)


_BY_MODEL_REPO = {
    "Team-RAMEN/pana_nakatsuka_act_pick_joint16_augxx_s40k_20260730": PICK_LEG_ACT_EP2101_FRAME0,
    "Team-RAMEN/groot-n1.7-pick-legs-ver1": PICK_LEG_FRAME0,
    "Team-RAMEN/groot-n1.7-pick-legs-ver2-lora": PICK_LEG_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_takada_groot_n17_coarse_insert_100k_dex1_v2": COARSE_INSERT_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_1": FLIP_TABLE_V1_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_2": FLIP_TABLE_V2_FRAME0,
}


def subtask_start_pose_for_model(model_repo_id: str) -> SubtaskStartPose:
    """Resolve a pinned start pose, failing closed for unknown real models."""

    try:
        return _BY_MODEL_REPO[model_repo_id]
    except KeyError as exc:
        raise ValueError(
            "no verified dataset frame-zero arm pose is registered for "
            f"{model_repo_id!r}; physical actuation is refused"
        ) from exc
