"""Register flip-table residual RL configurations with RoboFinals."""

import gymnasium as gym

from . import agents


def _register(name: str, class_name: str) -> None:
    env_id = f"Robocasa-Rl-{name}"
    if env_id in gym.registry:
        return
    gym.register(
        id=env_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": f"robofinals_rl.flip_table.flip_table:{class_name}",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        },
        disable_env_checker=True,
    )


_register("FlipTableResidualStateRL", "FlipTableResidualStateRL")
_register("FlipTableResidualVisualRL", "FlipTableResidualVisualRL")
