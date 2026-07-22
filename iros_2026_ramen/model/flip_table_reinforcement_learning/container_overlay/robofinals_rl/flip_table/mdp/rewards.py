"""Privileged training rewards for flip-table RL.

These functions may read simulator object/contact state, but none of those
values are exposed through the actor or critic observation configuration.
"""

from __future__ import annotations

import os

import torch
from isaaclab.utils.math import matrix_from_quat
from robofinals.utils.isaac_data_compat import as_torch

from ..common import DEFAULT_STAGE


FINGER_BODY_NAMES = (
    "left_dex1_finger_link_1",
    "left_dex1_finger_link_2",
    "right_dex1_finger_link_1",
    "right_dex1_finger_link_2",
)

# Collision-mesh centroids from the organizer Dex1 URDF/STL, in each finger
# link frame. body_pos_w is the prismatic-joint origin and is about 10 cm
# behind the actual contact region.
FINGER_CONTACT_LOCAL_OFFSETS_M = (
    (0.10600786, -0.02882335, 0.0),
    (0.10686015, 0.02870791, 0.0),
    (0.10600786, -0.02882335, 0.0),
    (0.10686015, 0.02870791, 0.0),
)

# A valid grasp combines same-leg geometry, opposing force, and actuator
# engagement. The shaft is about 43.8 mm square, while the
# measured open Dex1 contact-center separation is about 106 mm. Exact opposing
# leg-body forces are geometrically restricted to the shaft segment; closure
# only rejects a fully open hand that happens to touch it.
GRASP_SUCCESS_THRESHOLD = 0.55
DEX1_GRASP_ENGAGEMENT_START = 0.005
DEX1_GRASP_ENGAGEMENT_FULL = 0.025
# Conservative simulator acceptance gate, not a published Dex1-1 hardware
# rating.  Keep this fixed for comparable offline experiments; real deployment
# must use independently validated actuator/current/torque/contact safeguards.
MAX_SAFE_FINGER_FORCE_N = 15.0
WHITE_TABLE_LEG_COUNT = 4
WHITE_LEG_CONTACT_SENSOR_NAMES = tuple(
    f"white_leg_contact_{index}" for index in range(WHITE_TABLE_LEG_COUNT)
)
LEG_CONTACT_CENTERLINE_DISTANCE_M = 0.055
LEG_CONTACT_ENDPOINT_MARGIN_M = 0.025
LEG_CONTACT_MIN_RADIAL_FORCE_FRACTION = 0.55


def _task(env):
    return env.cfg.isaaclab_arena_env.task


def contact_measurements_ready_from_steps(
    episode_steps: torch.Tensor,
    warmup_steps: int = 4,
) -> torch.Tensor:
    """Return which environments have advanced beyond contact sensor warm-up."""

    if warmup_steps < 0:
        raise ValueError("contact sensor warmup_steps must be non-negative")
    return episode_steps.long() > warmup_steps


def _contact_measurements_ready(env) -> torch.Tensor:
    """Mask PhysX contact-cache transients immediately after an environment reset."""

    warmup_steps = int(os.environ.get("FLIP_TABLE_RL_CONTACT_SENSOR_WARMUP_STEPS", "4"))
    episode_steps = as_torch(env.episode_length_buf).long()
    return contact_measurements_ready_from_steps(episode_steps, warmup_steps)


def _table_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    pos, quat = _task(env)._table_body_pose(env)
    if pos is None or quat is None:
        raise RuntimeError("flip-table body pose is unavailable")
    return pos, quat


def _finger_positions(env) -> torch.Tensor:
    """Return left/right pairs of real Dex1 finger-body centers."""

    robot = env.scene["robot"]
    body_ids = getattr(env, "_flip_table_rl_finger_body_ids", None)
    if body_ids is None:
        body_ids, resolved = robot.find_bodies(list(FINGER_BODY_NAMES), preserve_order=True)
        if tuple(resolved) != FINGER_BODY_NAMES:
            raise RuntimeError(f"unexpected Dex1 finger body order: {resolved}")
        env._flip_table_rl_finger_body_ids = body_ids
    body_pos = as_torch(robot.data.body_pos_w)[:, body_ids, :3]
    body_quat = as_torch(robot.data.body_quat_w)[:, body_ids, :4]
    rotation = matrix_from_quat(body_quat)
    local_offsets = torch.tensor(
        FINGER_CONTACT_LOCAL_OFFSETS_M,
        device=body_pos.device,
        dtype=body_pos.dtype,
    )
    world_offsets = torch.matmul(
        rotation,
        local_offsets.view(1, 4, 3, 1),
    ).squeeze(-1)
    fingers = body_pos + world_offsets
    return torch.stack((fingers[:, 0:2], fingers[:, 2:4]), dim=1)


def _hand_positions(env) -> torch.Tensor:
    """Return the center between each pair of real Dex1 fingers."""

    return _finger_positions(env).mean(dim=2)


def hand_positions(env) -> torch.Tensor:
    """Return real-observable hand centers for privileged reward shaping."""

    return _hand_positions(env)


def finger_positions(env) -> torch.Tensor:
    """Return collision-centroid finger positions for teacher diagnostics."""

    return _finger_positions(env)


def leg_positions(env) -> torch.Tensor:
    """Return white-table leg centers for privileged teacher diagnostics."""

    return _leg_positions(env)


def _leg_positions(env) -> torch.Tensor:
    values = []
    task = _task(env)
    for name, prim_path, _site_path in task.leg_reg_int_sites:
        try:
            value = as_torch(env.scene[name].data.root_pos_w)[:, :3]
        except KeyError:
            value, _quat = task._extract_object_pose(env, name, prim_path)
            if value is None:
                raise RuntimeError(f"white-table leg pose is unavailable: {name}")
        values.append(value[:, :3])
    return torch.stack(values, dim=1)


def distinct_bimanual_segment_distances(
    hands: torch.Tensor,
    centers: torch.Tensor,
    axes: torch.Tensor,
    half_length: float,
) -> torch.Tensor:
    """Assign each hand to a distinct finite leg-axis segment."""

    if hands.ndim != 3 or hands.shape[1:] != (2, 3):
        raise ValueError(f"hands must be [B, 2, 3], got {tuple(hands.shape)}")
    if centers.ndim != 3 or centers.shape[0] != hands.shape[0] or centers.shape[2] != 3:
        raise ValueError(f"centers must be [B, K, 3], got {tuple(centers.shape)}")
    if axes.shape != (hands.shape[0], 3):
        raise ValueError(f"axes must be [B, 3], got {tuple(axes.shape)}")
    if half_length <= 0:
        raise ValueError("half_length must be positive")

    unit_axes = torch.nn.functional.normalize(axes, dim=-1)
    relative = hands[:, :, None, :] - centers[:, None, :, :]
    along = torch.sum(relative * unit_axes[:, None, None, :], dim=-1).clamp(
        -half_length,
        half_length,
    )
    closest = centers[:, None, :, :] + along[..., None] * unit_axes[:, None, None, :]
    distances = torch.linalg.norm(hands[:, :, None, :] - closest, dim=-1)
    count = centers.shape[1]
    if count < 2:
        raise ValueError("at least two segments are required")
    assignments = [(left, right) for left in range(count) for right in range(count) if left != right]
    costs = torch.stack([distances[:, 0, left] + distances[:, 1, right] for left, right in assignments], dim=1)
    candidates = torch.stack(
        [torch.stack([distances[:, 0, left], distances[:, 1, right]], dim=1) for left, right in assignments],
        dim=1,
    )
    best = costs.argmin(dim=1)
    return candidates[torch.arange(hands.shape[0], device=hands.device), best]


def segment_distance_matrix(
    hands: torch.Tensor,
    centers: torch.Tensor,
    axes: torch.Tensor,
    half_length: float,
) -> torch.Tensor:
    """Return every hand-to-segment distance as a ``[B, 2, K]`` tensor."""

    if hands.ndim != 3 or hands.shape[1:] != (2, 3):
        raise ValueError(f"hands must be [B, 2, 3], got {tuple(hands.shape)}")
    if centers.ndim != 3 or centers.shape[0] != hands.shape[0] or centers.shape[2] != 3:
        raise ValueError(f"centers must be [B, K, 3], got {tuple(centers.shape)}")
    if axes.shape != (hands.shape[0], 3):
        raise ValueError(f"axes must be [B, 3], got {tuple(axes.shape)}")
    if half_length <= 0:
        raise ValueError("half_length must be positive")

    unit_axes = torch.nn.functional.normalize(axes, dim=-1)
    relative = hands[:, :, None, :] - centers[:, None, :, :]
    along = torch.sum(relative * unit_axes[:, None, None, :], dim=-1).clamp(
        -half_length,
        half_length,
    )
    closest = centers[:, None, :, :] + along[..., None] * unit_axes[:, None, None, :]
    return torch.linalg.norm(hands[:, :, None, :] - closest, dim=-1)


def nearest_segment_distances(
    hands: torch.Tensor,
    centers: torch.Tensor,
    axes: torch.Tensor,
    half_length: float,
) -> torch.Tensor:
    """Return each hand's distance to its nearest finite leg-axis segment."""

    return segment_distance_matrix(hands, centers, axes, half_length).amin(dim=2)


def filter_finger_forces_by_leg_segments(
    fingers: torch.Tensor,
    force_vectors_by_leg: torch.Tensor,
    centers: torch.Tensor,
    axes: torch.Tensor,
    half_length: float,
    *,
    endpoint_margin: float = LEG_CONTACT_ENDPOINT_MARGIN_M,
    distance_threshold: float = LEG_CONTACT_CENTERLINE_DISTANCE_M,
    min_radial_force_fraction: float = LEG_CONTACT_MIN_RADIAL_FORCE_FRACTION,
) -> torch.Tensor:
    """Validate exact PhysX filter forces against their corresponding leg shafts.

    ``force_vectors_by_leg`` preserves the contact sensor's filter-partner axis,
    so force reported for one leg can never be reassigned to a nearer different
    leg. The geometric checks reject shaft-end and predominantly axial contact.
    """

    if fingers.ndim != 4 or fingers.shape[1:] != (2, 2, 3):
        raise ValueError(f"fingers must be [B,2,2,3], got {tuple(fingers.shape)}")
    expected_force_shape = (fingers.shape[0], 2, 2, centers.shape[1], 3)
    if force_vectors_by_leg.shape != expected_force_shape:
        raise ValueError(
            f"force_vectors_by_leg must be {expected_force_shape}, "
            f"got {tuple(force_vectors_by_leg.shape)}"
        )
    if centers.ndim != 3 or centers.shape[0] != fingers.shape[0] or centers.shape[2] != 3:
        raise ValueError(f"centers must be [B,K,3], got {tuple(centers.shape)}")
    if axes.shape != (fingers.shape[0], 3):
        raise ValueError(f"axes must be [B,3], got {tuple(axes.shape)}")
    if centers.shape[1] != WHITE_TABLE_LEG_COUNT:
        raise ValueError(
            f"expected {WHITE_TABLE_LEG_COUNT} leg centers, got {centers.shape[1]}"
        )
    if half_length <= 0 or distance_threshold <= 0:
        raise ValueError("leg geometry dimensions must be positive")
    if not 0 <= endpoint_margin < half_length:
        raise ValueError("endpoint_margin must be in [0, half_length)")
    if not 0 <= min_radial_force_fraction <= 1:
        raise ValueError("min_radial_force_fraction must be in [0,1]")

    unit_axes = torch.nn.functional.normalize(axes, dim=-1)
    relative = fingers[:, :, :, None, :] - centers[:, None, None, :, :]
    along = torch.sum(relative * unit_axes[:, None, None, None, :], dim=-1)
    closest = centers[:, None, None, :, :] + along.clamp(
        -half_length,
        half_length,
    )[..., None] * unit_axes[:, None, None, None, :]
    distances = torch.linalg.norm(fingers[:, :, :, None, :] - closest, dim=-1)

    axial_force = torch.sum(
        force_vectors_by_leg * unit_axes[:, None, None, None, :],
        dim=-1,
    )
    radial_vectors = (
        force_vectors_by_leg
        - axial_force[..., None] * unit_axes[:, None, None, None, :]
    )
    radial_force = torch.linalg.norm(radial_vectors, dim=-1)
    total_force = torch.linalg.norm(force_vectors_by_leg, dim=-1)
    radial_fraction = radial_force / total_force.clamp_min(1.0e-6)
    valid = (
        (distances <= distance_threshold)
        & (along.abs() <= half_length - endpoint_margin)
        & (radial_fraction >= min_radial_force_fraction)
    )
    return radial_force * valid


def finger_pair_segment_alignment_costs(
    fingers: torch.Tensor,
    centers: torch.Tensor,
    axes: torch.Tensor,
    half_length: float,
) -> torch.Tensor:
    """Return every finger-pair/segment straddling cost as ``[B, 2, K]``."""

    if fingers.ndim != 4 or fingers.shape[1:] != (2, 2, 3):
        raise ValueError(f"fingers must be [B, 2, 2, 3], got {tuple(fingers.shape)}")
    if centers.ndim != 3 or centers.shape[0] != fingers.shape[0] or centers.shape[2] != 3:
        raise ValueError(f"centers must be [B, K, 3], got {tuple(centers.shape)}")
    if axes.shape != (fingers.shape[0], 3):
        raise ValueError(f"axes must be [B, 3], got {tuple(axes.shape)}")
    if half_length <= 0:
        raise ValueError("half_length must be positive")

    unit_axes = torch.nn.functional.normalize(axes, dim=-1)
    midpoint = fingers.mean(dim=2)
    relative_midpoint = midpoint[:, :, None, :] - centers[:, None, :, :]
    along = torch.sum(relative_midpoint * unit_axes[:, None, None, :], dim=-1)
    clamped_along = along.clamp(-half_length, half_length)
    closest = centers[:, None, :, :] + clamped_along[..., None] * unit_axes[:, None, None, :]

    offsets = fingers[:, :, :, None, :] - closest[:, :, None, :, :]
    axial = torch.sum(offsets * unit_axes[:, None, None, None, :], dim=-1, keepdim=True)
    perpendicular = offsets - axial * unit_axes[:, None, None, None, :]
    radii = torch.linalg.norm(perpendicular, dim=-1)
    midpoint_distance = torch.linalg.norm(perpendicular.mean(dim=2), dim=-1)
    balance = torch.abs(radii[:, :, 0, :] - radii[:, :, 1, :])
    dot = torch.sum(perpendicular[:, :, 0, :] * perpendicular[:, :, 1, :], dim=-1)
    cosine = dot / (radii[:, :, 0, :] * radii[:, :, 1, :]).clamp_min(1.0e-6)
    same_side_penalty = 0.05 * torch.relu(cosine)
    axial_excess = torch.abs(along - clamped_along)
    return midpoint_distance + 0.5 * balance + same_side_penalty + axial_excess


def finger_pair_segment_alignment_cost(
    fingers: torch.Tensor,
    centers: torch.Tensor,
    axes: torch.Tensor,
    half_length: float,
) -> torch.Tensor:
    """Cost for each finger pair to straddle its best finite leg segment."""

    return finger_pair_segment_alignment_costs(fingers, centers, axes, half_length).amin(dim=2)


def _finger_contact_forces_for_side(env, side: str) -> torch.Tensor:
    """Return all-surface force magnitudes for the two fingers on one hand."""

    magnitudes = []
    for suffix in ("", "_2"):
        sensor = env.scene.sensors[f"{side}_gripper_contact{suffix}"]
        values = getattr(sensor.data, "net_forces_w")
        force = as_torch(values)
        norm = torch.linalg.norm(force[..., :3], dim=-1)
        magnitudes.append(norm.reshape(norm.shape[0], -1).amax(dim=1))
    return torch.stack(magnitudes, dim=1)


def _finger_contact_force_vectors_for_side(env, side: str) -> torch.Tensor:
    """Return strongest all-surface force vectors for one hand's fingers."""

    vectors = []
    for suffix in ("", "_2"):
        sensor = env.scene.sensors[f"{side}_gripper_contact{suffix}"]
        values = getattr(sensor.data, "net_forces_w")
        force = as_torch(values)[..., :3].reshape(env.num_envs, -1, 3)
        strongest = torch.linalg.norm(force, dim=-1).argmax(dim=1)
        batch_ids = torch.arange(env.num_envs, device=force.device)
        vectors.append(force[batch_ids, strongest])
    return torch.stack(vectors, dim=1)


def _finger_white_leg_force_vectors_by_leg_for_side(env, side: str) -> torch.Tensor:
    """Return per-leg forces using GPU-supported reverse body filters.

    Each sensor measures the force on one leg body from all four Dex1 finger
    bodies. Negation converts it to force on the fingers, preserving the
    convention used by the all-surface finger sensors.
    """

    vectors_by_leg = []
    for sensor_name in WHITE_LEG_CONTACT_SENSOR_NAMES:
        sensor = env.scene.sensors[sensor_name]
        values = getattr(sensor.data, "force_matrix_w", None)
        if values is None:
            raise RuntimeError(
                f"{sensor_name} has no filtered contact-force matrix; "
                "the reverse white-leg contact contract is not active"
            )
        force = as_torch(values)[..., :3]
        if force.ndim != 4 or force.shape[0] != env.num_envs or force.shape[1] != 1:
            raise RuntimeError(
                f"{sensor_name} has unexpected filtered force shape {tuple(force.shape)}; "
                "expected [num_envs, 1, num_fingers, 3]"
            )
        force = force[:, 0]
        if force.shape[1] != len(FINGER_BODY_NAMES):
            raise RuntimeError(f"{sensor_name} resolved no Dex1 finger contact partners")
        vectors_by_leg.append(-force)

    # [B, four fingers, four legs, xyz], then select one hand's finger pair.
    all_fingers = torch.stack(vectors_by_leg, dim=2)
    finger_slice = slice(0, 2) if side == "left" else slice(2, 4)
    return all_fingers[:, finger_slice]


def bimanual_reach(env, std: float = 0.18) -> torch.Tensor:
    distances = bimanual_reach_distances(env)
    per_hand = torch.exp(-distances / std)
    # A pure mean lets the easier hand dominate. Weight the worse hand so both
    # Dex1 grippers must approach two distinct legs before reward saturates.
    return 0.25 * per_hand.mean(dim=1) + 0.75 * per_hand.amin(dim=1)


def reach_gate_from_distances(
    distances: torch.Tensor,
    threshold: float = 0.17,
    margin: float = 0.04,
) -> torch.Tensor:
    """Shape the worst-hand distance immediately outside the reach gate."""

    if distances.ndim != 2 or distances.shape[1] != 2:
        raise ValueError(f"distances must be [B, 2], got {tuple(distances.shape)}")
    if margin <= 0:
        raise ValueError("margin must be positive")
    excess = torch.relu(distances.amax(dim=1) - threshold)
    return torch.exp(-excess / margin)


def bimanual_reach_gate(env, threshold: float = 0.17, margin: float = 0.04) -> torch.Tensor:
    """Dense reward for moving both hands through the unchanged success gate."""

    return reach_gate_from_distances(
        bimanual_reach_distances(env),
        threshold=threshold,
        margin=margin,
    )


def bimanual_reach_synchrony(
    env,
    threshold: float = 0.10,
    margin: float = 0.05,
    balance_std: float = 0.03,
) -> torch.Tensor:
    """Reward both hands being near their legs at the same control step."""

    if balance_std <= 0:
        raise ValueError("balance_std must be positive")
    distances = bimanual_reach_distances(env)
    gate = reach_gate_from_distances(distances, threshold=threshold, margin=margin)
    balance = torch.exp(-torch.abs(distances[:, 0] - distances[:, 1]) / balance_std)
    return gate * balance


def bimanual_reach_distances(env) -> torch.Tensor:
    """Distance from each Dex1 center to two distinct finite leg axes."""

    _table_pos, table_quat = _table_pose(env)
    table_normal = matrix_from_quat(table_quat)[:, :, 2]
    half_length = float(os.environ.get("FLIP_TABLE_RL_LEG_GRASP_HALF_LENGTH_M", "0.16"))
    return distinct_bimanual_segment_distances(
        _hand_positions(env),
        _leg_positions(env),
        table_normal,
        half_length,
    )


def hand_leg_distances(env) -> torch.Tensor:
    """Distance from each hand to every white-table leg segment."""

    _table_pos, table_quat = _table_pose(env)
    table_normal = matrix_from_quat(table_quat)[:, :, 2]
    half_length = float(os.environ.get("FLIP_TABLE_RL_LEG_GRASP_HALF_LENGTH_M", "0.16"))
    return segment_distance_matrix(
        _hand_positions(env),
        _leg_positions(env),
        table_normal,
        half_length,
    )


def nearest_leg_distances(env) -> torch.Tensor:
    """Distance from each hand to any leg, for sequential hand curricula."""

    return hand_leg_distances(env).amin(dim=2)


def finger_leg_alignment_costs(env) -> torch.Tensor:
    """Cost for each hand's finger pair to straddle every leg shaft."""

    _table_pos, table_quat = _table_pose(env)
    table_normal = matrix_from_quat(table_quat)[:, :, 2]
    half_length = float(os.environ.get("FLIP_TABLE_RL_LEG_GRASP_HALF_LENGTH_M", "0.16"))
    return finger_pair_segment_alignment_costs(
        _finger_positions(env),
        _leg_positions(env),
        table_normal,
        half_length,
    )


def finger_leg_alignment_cost(env) -> torch.Tensor:
    """Per-hand cost for placing two fingers around one leg shaft."""

    return finger_leg_alignment_costs(env).amin(dim=2)


def same_leg_grasp_geometry(
    distances: torch.Tensor,
    alignment_costs: torch.Tensor,
    *,
    distance_threshold: float = 0.04,
    alignment_threshold: float = 0.025,
) -> torch.Tensor:
    """Return per-leg masks for a finger pair that actually straddles one leg.

    The distance and alignment conditions must hold for the same leg.  Taking
    their minima independently can incorrectly combine proximity to one leg
    with finger alignment around another leg.
    """

    if distances.ndim != 3 or distances.shape[1] != 2:
        raise ValueError(f"distances must be [B, 2, K], got {tuple(distances.shape)}")
    if alignment_costs.shape != distances.shape:
        raise ValueError(
            "alignment_costs must have the same [B, 2, K] shape as distances, "
            f"got {tuple(alignment_costs.shape)}"
        )
    if distance_threshold <= 0 or alignment_threshold <= 0:
        raise ValueError("grasp geometry thresholds must be positive")
    return (distances < distance_threshold) & (alignment_costs < alignment_threshold)


def contact_leg_mask(
    distances: torch.Tensor,
    finger_forces_by_leg: torch.Tensor,
    *,
    force_threshold: float = 0.5,
    distance_threshold: float = 0.06,
) -> torch.Tensor:
    """Return legs contacted by at least one nearby finger of one hand."""

    if distances.ndim != 2:
        raise ValueError(f"distances must be [B,K], got {tuple(distances.shape)}")
    expected = (distances.shape[0], 2, distances.shape[1])
    if finger_forces_by_leg.shape != expected:
        raise ValueError(
            f"finger_forces_by_leg must be {expected}, got {tuple(finger_forces_by_leg.shape)}"
        )
    if force_threshold <= 0 or distance_threshold <= 0:
        raise ValueError("contact thresholds must be positive")
    return (distances < distance_threshold) & (
        finger_forces_by_leg.amax(dim=1) >= force_threshold
    )


def distinct_bimanual_leg_quality(values: torch.Tensor) -> torch.Tensor:
    """Return best worst-hand quality while assigning distinct legs."""

    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"values must be [B,2,K], got {tuple(values.shape)}")
    count = values.shape[2]
    if count < 2:
        raise ValueError("at least two legs are required")
    candidates = [
        torch.minimum(values[:, 0, left], values[:, 1, right])
        for left in range(count)
        for right in range(count)
        if left != right
    ]
    return torch.stack(candidates, dim=1).amax(dim=1)


def bimanual_contact_forces(env) -> torch.Tensor:
    """Return left/right all-surface force magnitudes for safety diagnostics."""

    return finger_contact_forces(env).amax(dim=2)


def finger_contact_forces(env) -> torch.Tensor:
    """Return all-surface contact magnitude for both fingers of both hands."""

    return torch.stack(
        (
            _finger_contact_forces_for_side(env, "left"),
            _finger_contact_forces_for_side(env, "right"),
        ),
        dim=1,
    )


def finger_contact_force_vectors(env) -> torch.Tensor:
    """Return all-surface world-frame force vectors for all four fingers."""

    return torch.stack(
        (
            _finger_contact_force_vectors_for_side(env, "left"),
            _finger_contact_force_vectors_for_side(env, "right"),
        ),
        dim=1,
    )


def white_table_leg_contact_force_vectors(env) -> torch.Tensor:
    """Return exact white-leg-only force vectors for all four fingers."""

    return white_table_leg_contact_force_vectors_by_leg(env).sum(dim=3)


def white_table_leg_contact_force_vectors_by_leg(env) -> torch.Tensor:
    """Return exact filtered vectors as ``[B, hand, finger, leg, xyz]``."""

    return torch.stack(
        (
            _finger_white_leg_force_vectors_by_leg_for_side(env, "left"),
            _finger_white_leg_force_vectors_by_leg_for_side(env, "right"),
        ),
        dim=1,
    )


def white_table_leg_contact_forces(env) -> torch.Tensor:
    """Return attributed white-leg force magnitudes as ``[B, 2, 2, 4]``."""

    _table_pos, table_quat = _table_pose(env)
    table_normal = matrix_from_quat(table_quat)[:, :, 2]
    half_length = float(os.environ.get("FLIP_TABLE_RL_LEG_GRASP_HALF_LENGTH_M", "0.16"))
    forces = filter_finger_forces_by_leg_segments(
        _finger_positions(env),
        white_table_leg_contact_force_vectors_by_leg(env),
        _leg_positions(env),
        table_normal,
        half_length=half_length,
    )
    return forces * _contact_measurements_ready(env).to(forces).view(-1, 1, 1, 1)


def unsafe_finger_force_penalty_from_forces(
    forces: torch.Tensor,
    max_safe_force_n: float = MAX_SAFE_FINGER_FORCE_N,
) -> torch.Tensor:
    """Return normalized force above the real-hardware safety envelope."""

    if forces.ndim != 3 or forces.shape[1:] != (2, 2):
        raise ValueError(f"finger forces must be [B,2,2], got {tuple(forces.shape)}")
    if max_safe_force_n <= 0:
        raise ValueError("max_safe_force_n must be positive")
    maximum = forces.amax(dim=(1, 2))
    return torch.relu(maximum - max_safe_force_n) / max_safe_force_n


def finger_force_margin_penalty_from_forces(
    forces: torch.Tensor,
    margin_start_n: float = 5.0,
    max_safe_force_n: float = MAX_SAFE_FINGER_FORCE_N,
) -> torch.Tensor:
    """Penalize approaching the hard force limit before contact becomes unsafe."""

    if forces.ndim != 3 or forces.shape[1:] != (2, 2):
        raise ValueError(f"finger forces must be [B,2,2], got {tuple(forces.shape)}")
    if not 0 <= margin_start_n < max_safe_force_n:
        raise ValueError("force margin requires 0 <= start < max_safe_force_n")
    maximum = forces.amax(dim=(1, 2))
    return torch.relu(maximum - margin_start_n) / (max_safe_force_n - margin_start_n)


def finger_force_margin_penalty(
    env,
    margin_start_n: float = 5.0,
    max_safe_force_n: float = MAX_SAFE_FINGER_FORCE_N,
) -> torch.Tensor:
    penalty = finger_force_margin_penalty_from_forces(
        finger_contact_forces(env),
        margin_start_n=margin_start_n,
        max_safe_force_n=max_safe_force_n,
    )
    return penalty * _contact_measurements_ready(env).to(penalty)


def unsafe_finger_force_penalty(
    env,
    max_safe_force_n: float = MAX_SAFE_FINGER_FORCE_N,
) -> torch.Tensor:
    penalty = unsafe_finger_force_penalty_from_forces(
        finger_contact_forces(env),
        max_safe_force_n=max_safe_force_n,
    )
    return penalty * _contact_measurements_ready(env).to(penalty)


def finger_force_is_safe(
    env,
    max_safe_force_n: float = MAX_SAFE_FINGER_FORCE_N,
) -> torch.Tensor:
    raw_safe = finger_contact_forces(env).amax(dim=(1, 2)) <= max_safe_force_n
    return ~_contact_measurements_ready(env) | raw_safe


def single_hand_reach(env, side: str = "right", std: float = 0.12) -> torch.Tensor:
    """Dense reach reward for the demonstration's active hand."""

    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    if std <= 0:
        raise ValueError("std must be positive")
    index = 0 if side == "left" else 1
    return torch.exp(-nearest_leg_distances(env)[:, index] / std)


def single_hand_contact(
    env,
    side: str = "right",
    force_threshold: float = 0.5,
    distance_threshold: float = 0.06,
) -> torch.Tensor:
    """Contact gate requiring force against the same nearby white-table leg."""

    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    index = 0 if side == "left" else 1
    contacted = contact_leg_mask(
        hand_leg_distances(env)[:, index],
        white_table_leg_contact_forces(env)[:, index],
        force_threshold=force_threshold,
        distance_threshold=distance_threshold,
    )
    return contacted.any(dim=1).float()


def _finger_closure(env) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids = getattr(env, "_flip_table_rl_grasp_finger_joint_ids", None)
    if joint_ids is None:
        names = (
            "left_dex1_finger_joint_1",
            "left_dex1_finger_joint_2",
            "right_dex1_finger_joint_1",
            "right_dex1_finger_joint_2",
        )
        joint_ids, resolved = robot.find_joints(list(names), preserve_order=True)
        if tuple(resolved) != names:
            raise RuntimeError(f"unexpected Dex1 joint order: {resolved}")
        env._flip_table_rl_grasp_finger_joint_ids = joint_ids
    fingers = as_torch(robot.data.joint_pos)[:, joint_ids]
    closure = ((0.0245 - fingers) / (0.0245 - (-0.02))).clamp(0.0, 1.0)
    return torch.stack((closure[:, 0:2].mean(dim=1), closure[:, 2:4].mean(dim=1)), dim=1)


def finger_closure(env) -> torch.Tensor:
    """Return normalized left/right Dex1 closure for diagnostics."""

    return _finger_closure(env)


def grasp_closure_engagement(
    closure: torch.Tensor,
    start: float = 0.005,
    full: float = 0.025,
) -> torch.Tensor:
    """Map normalized Dex1 closure to thick-object engagement."""

    if start < 0 or full <= start:
        raise ValueError("grasp engagement requires 0 <= start < full")
    return ((closure - start) / (full - start)).clamp(0.0, 1.0)


def single_hand_grasp_by_leg(
    env,
    side: str = "right",
    force_threshold: float = 0.5,
    distance_threshold: float = 0.04,
    alignment_threshold: float = 0.025,
) -> torch.Tensor:
    """Return strict Dex1 grasp quality for every white-table leg."""

    if side not in {"left", "right"}:
        raise ValueError(f"side must be left or right, got {side!r}")
    index = 0 if side == "left" else 1
    geometry = same_leg_grasp_geometry(
        hand_leg_distances(env),
        finger_leg_alignment_costs(env),
        distance_threshold=distance_threshold,
        alignment_threshold=alignment_threshold,
    )[:, index]
    forces = white_table_leg_contact_forces(env)[:, index]
    opposing_contact = (forces >= force_threshold).all(dim=1)
    force_quality = torch.tanh(forces.amin(dim=1) / 2.0)
    closure = _finger_closure(env)[:, index].unsqueeze(1)
    engagement = grasp_closure_engagement(
        closure,
        start=DEX1_GRASP_ENGAGEMENT_START,
        full=DEX1_GRASP_ENGAGEMENT_FULL,
    )
    return geometry.float() * opposing_contact.float() * force_quality * engagement


def single_hand_grasp_for_leg(
    env,
    target_leg: torch.Tensor,
    side: str = "right",
) -> torch.Tensor:
    """Return strict grasp quality for one explicitly selected leg per env."""

    values = single_hand_grasp_by_leg(env, side=side)
    target = torch.as_tensor(target_leg, device=values.device, dtype=torch.long)
    if target.ndim == 0:
        target = target.expand(values.shape[0])
    if target.shape != (values.shape[0],):
        raise ValueError(f"target_leg must be scalar or [B], got {tuple(target.shape)}")
    if bool(torch.any((target < 0) | (target >= values.shape[1]))):
        raise ValueError("target_leg contains an out-of-range leg index")
    return values.gather(1, target.unsqueeze(1)).squeeze(1)


def single_hand_grasp(env, side: str = "right") -> torch.Tensor:
    """Dex1 closure rewarded only for a force-bearing same-leg straddle."""

    return single_hand_grasp_by_leg(env, side=side).amax(dim=1)


def bimanual_contact(
    env,
    force_threshold: float = 0.5,
    distance_threshold: float = 0.06,
) -> torch.Tensor:
    distances = hand_leg_distances(env)
    forces = white_table_leg_contact_forces(env)
    per_hand_leg = torch.stack(
        (
            contact_leg_mask(
                distances[:, 0],
                forces[:, 0],
                force_threshold=force_threshold,
                distance_threshold=distance_threshold,
            ),
            contact_leg_mask(
                distances[:, 1],
                forces[:, 1],
                force_threshold=force_threshold,
                distance_threshold=distance_threshold,
            ),
        ),
        dim=1,
    )
    left = per_hand_leg[:, 0].any(dim=1)
    right = per_hand_leg[:, 1].any(dim=1)
    distinct = distinct_bimanual_leg_quality(per_hand_leg.float()) >= 1.0
    return 0.25 * left.float() + 0.25 * right.float() + 0.5 * distinct.float()


def bimanual_grasp(env) -> torch.Tensor:
    """Reward closed Dex1 fingers only when both hands contact distinct legs."""

    return distinct_bimanual_leg_quality(
        torch.stack(
            (
                single_hand_grasp_by_leg(env, side="left"),
                single_hand_grasp_by_leg(env, side="right"),
            ),
            dim=1,
        )
    )


def table_lift_progress(env, target_height: float = 0.18) -> torch.Tensor:
    table_pos, _ = _table_pose(env)
    initial = getattr(_task(env), "_initial_table_pos", None)
    if initial is None:
        return torch.zeros(env.num_envs, device=env.device)
    gain = table_pos[:, 2] - initial.to(table_pos)[:, 2]
    return (gain / target_height).clamp(0.0, 1.0)


def grasped_table_lift(env, side: str = "right") -> torch.Tensor:
    """Reward lift only while a real Dex1 remains physically coupled to a leg."""

    return table_lift_progress(env) * single_hand_grasp(env, side=side)


def bimanual_grasped_table_lift(env) -> torch.Tensor:
    """Reward lift only while both Dex1 hands grasp distinct white-table legs."""

    return table_lift_progress(env) * bimanual_grasp(env)


def table_flip_progress(env) -> torch.Tensor:
    _, table_quat = _table_pose(env)
    initial = getattr(_task(env), "_initial_table_normal", None)
    if initial is None:
        return torch.zeros(env.num_envs, device=env.device)
    normal = matrix_from_quat(table_quat)[:, :, 2]
    dot = torch.sum(normal * initial.to(normal), dim=-1).clamp(-1.0, 1.0)
    return ((1.0 - dot) * 0.5).clamp(0.0, 1.0)


def table_disturbance_penalty(
    env,
    allowed_lift_m: float = 0.01,
    allowed_flip_progress: float = 0.03,
) -> torch.Tensor:
    """Penalize moving the table before the curriculum permits lifting it."""

    if allowed_lift_m < 0 or not 0.0 <= allowed_flip_progress < 1.0:
        raise ValueError("table disturbance thresholds are invalid")
    table_pos, _ = _table_pose(env)
    initial = getattr(_task(env), "_initial_table_pos", None)
    if initial is None:
        return torch.zeros(env.num_envs, device=env.device)
    lift_m = torch.relu(table_pos[:, 2] - initial.to(table_pos)[:, 2])
    lift_excess = torch.relu(lift_m - allowed_lift_m) / max(0.18 - allowed_lift_m, 1.0e-6)
    flip_excess = torch.relu(table_flip_progress(env) - allowed_flip_progress) / (
        1.0 - allowed_flip_progress
    )
    return lift_excess + flip_excess


def early_stage_table_is_safe(
    env,
    max_lift_m: float,
    max_flip_progress: float,
) -> torch.Tensor:
    if max_lift_m < 0 or not 0.0 <= max_flip_progress <= 1.0:
        raise ValueError("early-stage table safety thresholds are invalid")
    table_pos, _ = _table_pose(env)
    initial = getattr(_task(env), "_initial_table_pos", None)
    if initial is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    lift_m = table_pos[:, 2] - initial.to(table_pos)[:, 2]
    return (lift_m <= max_lift_m) & (table_flip_progress(env) <= max_flip_progress)


def table_stable_success(
    env,
    stable_steps: int | None = None,
) -> torch.Tensor:
    episode_steps = as_torch(env.episode_length_buf).long()
    last_steps = getattr(env, "_flip_table_rl_stable_last_steps", None)
    cached = getattr(env, "_flip_table_rl_stable_result", None)
    if last_steps is not None and cached is not None and torch.equal(last_steps, episode_steps):
        return cached.clone()

    task = _task(env)
    components = task._stable_flip_success_components(env)
    candidate = components["candidate"]
    if stable_steps is None:
        stable_steps = int(os.environ.get("FLIP_TABLE_SUCCESS_HOLD_STEPS", "20"))
    if stable_steps < 1:
        raise ValueError("stable_steps must be positive")
    streak = getattr(
        env,
        "_flip_table_rl_stable_streak",
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
    reset = episode_steps <= 1
    streak = torch.where(reset, torch.zeros_like(streak), torch.where(candidate, streak + 1, torch.zeros_like(streak)))
    result = candidate & (streak >= max(1, int(stable_steps)))

    env._flip_table_rl_stable_streak = streak
    env._flip_table_rl_stable_last_steps = episode_steps.clone()
    env._flip_table_rl_stable_result = result.clone()
    return result


def table_stage_success(env) -> torch.Tensor:
    """Return the explicit curriculum gate for the active training stage."""

    # The RL registration replaces the organizer task's ``_check_success``
    # termination, which normally enforces this lock after every physics step.
    # Preserve the fixed-lower-body evaluation contract on the replacement
    # path before observations for the next control tick are produced.
    task = _task(env)
    lock = getattr(task, "_apply_lower_body_lock", None)
    if lock is None:
        raise RuntimeError("flip-table task does not expose the lower-body lock")
    lock(env)

    stage = os.environ.get("FLIP_TABLE_RL_STAGE", DEFAULT_STAGE).strip().lower()
    force_safe = finger_force_is_safe(env)
    if stage == "reach":
        threshold = float(os.environ.get("FLIP_TABLE_RL_REACH_SUCCESS_DISTANCE_M", "0.08"))
        return (
            (nearest_leg_distances(env)[:, 1] <= threshold)
            & force_safe
            & early_stage_table_is_safe(env, max_lift_m=0.01, max_flip_progress=0.03)
        )
    if stage == "contact":
        return (
            (single_hand_contact(env, side="right") >= 1.0)
            & force_safe
            & early_stage_table_is_safe(env, max_lift_m=0.02, max_flip_progress=0.08)
        )
    if stage == "grasp":
        return (
            (single_hand_grasp(env, side="right") >= GRASP_SUCCESS_THRESHOLD)
            & force_safe
            & early_stage_table_is_safe(env, max_lift_m=0.04, max_flip_progress=0.25)
        )
    if stage == "sequential_lift":
        return (
            (table_lift_progress(env) >= 0.50)
            & (single_hand_grasp(env, side="right") >= GRASP_SUCCESS_THRESHOLD)
            & force_safe
        )
    if stage == "lift":
        # ``sequential_lift`` validates the demonstrated right-first raise.
        # This next stage is the measured left-hand join: a height-only gate
        # would otherwise mark the existing right-only prefix as successful.
        return (
            (table_lift_progress(env) >= 0.50)
            & (bimanual_grasp(env) >= GRASP_SUCCESS_THRESHOLD)
            & force_safe
        )
    if stage == "rotate":
        # Rotation is not accepted until the left hand has joined the
        # right-first lift on a distinct leg. This is a simulator-only success
        # predicate, never an actor or critic input.
        return (
            (table_flip_progress(env) >= 0.50)
            & (bimanual_grasp(env) >= GRASP_SUCCESS_THRESHOLD)
            & force_safe
        )
    if stage == "flip":
        return (table_flip_progress(env) >= 0.85) & force_safe
    if stage in {"stabilize", "full"}:
        return table_stable_success(env)
    raise ValueError(f"unknown FLIP_TABLE_RL_STAGE={stage!r}")


def stage_success_bonus(env) -> torch.Tensor:
    """Emit one success bonus per environment and episode."""

    success = table_stage_success(env)
    episode_steps = as_torch(env.episode_length_buf).long()
    seen = getattr(
        env,
        "_flip_table_rl_stage_success_seen",
        torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    seen = torch.where(episode_steps <= 1, torch.zeros_like(seen), seen)
    bonus = success & ~seen
    env._flip_table_rl_stage_success_seen = seen | success
    return bonus.float()


def demo_residual_l2(env) -> torch.Tensor:
    action = as_torch(env.action_manager.action)
    return torch.mean(action[:, :17] ** 2, dim=1)
