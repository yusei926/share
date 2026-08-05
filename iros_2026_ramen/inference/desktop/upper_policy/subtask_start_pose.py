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
    # Median action.hand_cmd at frame 0 over the same 2,114 episodes used for
    # the arm statistic above, converted from the dataset's 0..4.5 scalar to
    # the physical Dex1 opening fraction.  In particular, the left hand is
    # intentionally only about 46% open at the training start state.
    dex1_opening_fraction=(
        2.073647975921631 / 4.5,
        4.470532655715942 / 4.5,
    ),
    statistic=(
        "median_action_robot_q_desired_frame0_arms14;"
        "median_action_hand_cmd_frame0_dex1_2"
    ),
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
    # Exact deployment values recorded with the selected episode-2101
    # frame-zero pose.  Keeping these on the pose contract (instead of in the
    # ACT runner) lets every ACT subtask select its own training-start hands.
    dex1_opening_fraction=(4.43 / 4.5, 4.47 / 4.5),
    statistic=(
        "deployment_medoid_episode2101_frame0_action_arms14_and_dex1_2;"
        "source_commit=817e8addd944b43b9ada6d096feaa93f75179d38"
    ),
)

# Reconstructed from exactly the 227 episode indices recorded in both
# pre-straddle ACT checkpoints.  The public dataset contains 256 episodes, so
# using all episodes here would silently choose a different start distribution
# from the one actually used for training.
PRE_STRADDLE_ACT_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        -0.38675957918167114,
        0.6175763010978699,
        -0.3497064709663391,
        0.14622721076011658,
        -0.01363319531083107,
        0.228810653090477,
        -1.060666561126709,
        -0.14246605336666107,
        -0.6233853101730347,
        0.21238403022289276,
        -0.3058239817619324,
        0.2979549765586853,
        0.5277582406997681,
        0.9015522003173828,
    ),
    dataset_repo_id="Team-RAMEN/pana_nakatsuka_ikea_pre_straddle",
    dataset_revision="dd0059983d7149121793bb13f1718d54007287da",
    training_episode_count=227,
    statistic=(
        "median_action_frame0_arms14_and_dex1_2;"
        "episode_indices_from_checkpoint_train_config;"
        "reconstruction_sha256="
        "a5669a3bd5568d1ce2124139edf6b5713fecf522614729809b1b9c03938ec3b4"
    ),
    # The checkpoint names a local path and revision "main".  The supplied HF
    # dataset reproduces all selected rows, but was not pinned by the trainer.
    exact_training_revision=False,
    dex1_opening_fraction=(
        0.25341740250587463 / 4.5,
        1.0,
    ),
)

COARSE_INSERT_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        -0.5591480135917664, 0.4370526969432831, -0.35026293992996216,
        0.3723258972167969, -0.05962152034044266, 0.20113125443458557,
        -1.0550432205200195, -0.338015079498291, -0.2881007790565491,
        0.018755313009023666, 0.0006471482338383794, 0.3515812158584595,
        0.500892698764801, 0.5877543687820435,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_coarse_insert_1",
    dataset_revision="338c3db7f106e5108f29df5bb564a983897c0eca",
    training_episode_count=1697,
    # The model conditions on measured state, not desired action. Reconstruct
    # the start target from frame-zero observation rows in the exact 1,697
    # episode training split so the final pre-motion state matches the model's
    # input distribution. Physical Dex1 values are converted from 0..4.5.
    dex1_opening_fraction=(
        1.4682344198226929 / 4.5,
        2.2439992427825928 / 4.5,
    ),
    statistic=(
        "median_observation_robot_q_current_frame0_arms14;"
        "median_observation_hand_state_frame0_dex1_2;"
        "checkpoint_training_split_1697;split_seed=42"
    ),
    # The authoritative issue/94 Slurm launcher pins this exact dataset
    # revision even though the serialized train_config only records the local
    # materialized view.
    exact_training_revision=True,
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

# The baseline Furniture-GR00T candidate used only episode indices 0..155 from
# flip_table_2, whereas the finalized v2 deployment pose above uses all 171
# curated episodes.  Keep the candidate statistic separate so physical
# evaluation cannot silently start from a different training subset.  The
# checkpoint did not pin a dataset revision; this is a reproducible
# reconstruction from the currently sealed dataset snapshot, not a claim that
# the trainer consumed that exact Git revision.
FLIP_TABLE_GROOT_V2_BASELINE_TRAIN156_FRAME0 = SubtaskStartPose(
    arm_position_rad=(
        0.01607997575774789, 0.18264327943325043, 0.26085586845874786,
        -0.06220103055238724, -0.1442396268248558, 0.08705304563045502,
        -0.21875989437103271, -0.050612280145287514, -0.2726525217294693,
        0.0800039991736412, -0.07146013528108597, 0.32281383872032166,
        0.2702529579401016, 0.6890287697315216,
    ),
    dataset_repo_id="Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_2",
    dataset_revision="0dc47877dfb2efbea796a059c81290c649bc773c",
    training_episode_count=156,
    dex1_opening_fraction=(0.9225202136569552, 1.0),
    statistic=(
        "median_action_robot_q_desired_frame0_arms14;"
        "median_action_hand_cmd_frame0_dex1_2;"
        "episode_indices_0_through_155_from_checkpoint_train_config;"
        "reconstruction_sha256="
        "df3f888c8d8be373884427532d5b41c04d39aaac0d717d63024c75735844fb8c"
    ),
    exact_training_revision=False,
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
    "Team-RAMEN/pana_nakatsuka_act_pre_straddle_noaug_s20k_20260803": PRE_STRADDLE_ACT_FRAME0,
    "Team-RAMEN/pana_nakatsuka_act_pre_straddle_augxx_s40k_20260803": PRE_STRADDLE_ACT_FRAME0,
    "Team-RAMEN/groot-n1.7-pick-legs-ver1": PICK_LEG_FRAME0,
    "Team-RAMEN/groot-n1.7-pick-legs-ver2-lora": PICK_LEG_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_takada_groot_n17_coarse_insert_100k_dex1_v2": COARSE_INSERT_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_1": FLIP_TABLE_V1_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_diffusion_chunk_relative_2": FLIP_TABLE_V2_FRAME0,
    "Team-RAMEN/IROS2026_RAMEN_suzuki_flip_table_groot_n17_2_baseline_checkpoints": FLIP_TABLE_GROOT_V2_BASELINE_TRAIN156_FRAME0,
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
