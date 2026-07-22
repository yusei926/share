#!/usr/bin/env python3
"""Add an encoder-based joint continuity filter to the organizer G1 WBC action."""

from __future__ import annotations

import os
from pathlib import Path


MARKER = "FLIP_TABLE_WBC_JOINT_CONTINUITY_FILTER_V1"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"G1 WBC source changed; missing {label}")
    return text.replace(old, new, 1)


def main() -> None:
    root = Path(os.environ.get("ROBOFINALS_ROOT", "/workspace/robofinals"))
    path = root / "robofinals/core/mdp/actions/g1_action.py"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"[flip_table] G1 WBC continuity filter already present: {path}")
        return

    text = _replace_once(
        text,
        "from dataclasses import MISSING\n",
        "from dataclasses import MISSING\nimport os\n",
        "os import",
    )
    text = _replace_once(
        text,
        "        self._target_robot_joints_mujoco = None\n",
        "        self._target_robot_joints_mujoco = None\n"
        f"        # {MARKER}\n"
        "        self._continuity_filter_enabled = os.environ.get(\n"
        "            \"FLIP_TABLE_WBC_JOINT_CONTINUITY_FILTER\", \"false\"\n"
        "        ).strip().lower() in {\"1\", \"true\", \"yes\", \"on\"}\n"
        "        self._max_joint_speed_rad_s = float(os.environ.get(\n"
        "            \"FLIP_TABLE_WBC_MAX_JOINT_SPEED_RAD_S\", \"2.0\"\n"
        "        ))\n"
        "        self._max_joint_acceleration_rad_s2 = float(os.environ.get(\n"
        "            \"FLIP_TABLE_WBC_MAX_JOINT_ACCELERATION_RAD_S2\", \"8.0\"\n"
        "        ))\n"
        "        self._control_dt = float(env.cfg.sim.dt * env.cfg.decimation)\n"
        "        if self._continuity_filter_enabled and min(\n"
        "            self._max_joint_speed_rad_s,\n"
        "            self._max_joint_acceleration_rad_s2,\n"
        "            self._control_dt,\n"
        "        ) <= 0.0:\n"
        "            raise ValueError(\"G1 WBC continuity limits must be positive\")\n"
        "        self._continuity_target = None\n"
        "        self._continuity_velocity = torch.zeros(\n"
        "            (self.num_envs, self._num_joints), device=self.device\n"
        "        )\n",
        "continuity state initialization",
    )
    text = _replace_once(
        text,
        "        if self._negated_joint_indices:\n"
        "            self._processed_actions[:, self._negated_joint_indices] *= -1.0\n\n"
        "    def apply_actions(self):\n",
        "        if self._negated_joint_indices:\n"
        "            self._processed_actions[:, self._negated_joint_indices] *= -1.0\n"
        "        if self._continuity_filter_enabled:\n"
        "            measured = self._asset.data.joint_pos[:, self._joint_ids].clone()\n"
        "            if self._continuity_target is None:\n"
        "                self._continuity_target = measured\n"
        "                self._continuity_velocity.zero_()\n"
        "            error = self._processed_actions - self._continuity_target\n"
        "            stopping_speed = torch.sqrt(\n"
        "                2.0 * self._max_joint_acceleration_rad_s2 * torch.abs(error)\n"
        "            )\n"
        "            desired_velocity = torch.sign(error) * torch.minimum(\n"
        "                stopping_speed,\n"
        "                torch.full_like(stopping_speed, self._max_joint_speed_rad_s),\n"
        "            )\n"
        "            max_velocity_delta = (\n"
        "                self._max_joint_acceleration_rad_s2 * self._control_dt\n"
        "            )\n"
        "            velocity = self._continuity_velocity + torch.clamp(\n"
        "                desired_velocity - self._continuity_velocity,\n"
        "                -max_velocity_delta,\n"
        "                max_velocity_delta,\n"
        "            )\n"
        "            increment = velocity * self._control_dt\n"
        "            reached = torch.abs(increment) >= torch.abs(error)\n"
        "            self._continuity_target = torch.where(\n"
        "                reached, self._processed_actions, self._continuity_target + increment\n"
        "            )\n"
        "            self._continuity_velocity = torch.where(\n"
        "                reached, torch.zeros_like(velocity), velocity\n"
        "            )\n"
        "            self._processed_actions = self._continuity_target.clone()\n\n"
        "    def apply_actions(self):\n",
        "continuity filter",
    )
    text = _replace_once(
        text,
        "        self.upperbody_controller.body_ik_solver.reset()\n",
        "        self.upperbody_controller.body_ik_solver.reset()\n"
        "        self._continuity_target = None\n"
        "        self._continuity_velocity.zero_()\n",
        "continuity reset",
    )
    path.write_text(text, encoding="utf-8")
    print(f"[flip_table] added encoder-based G1 WBC joint continuity filter: {path}")


if __name__ == "__main__":
    main()
