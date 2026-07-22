from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model.flip_table_reinforcement_learning.rlpd import (
    PolicyControlClock,
    RLPDAgent,
    RLPDConfig,
    ReplayBuffer,
)
from model.flip_table_reinforcement_learning.rlpd.replay import balanced_replay_sample
from model.flip_table_reinforcement_learning.rlpd.policy_contract import (
    AbsoluteTargetDelayBuffer,
    BODY_RESIDUAL_SCALE,
    UpperBodyTargetSafetyFilter,
    apply_residual_to_base,
    stochastic_action_mask_for_stage,
    target_safety_from_environment,
)
from model.flip_table_reinforcement_learning.rlpd.sim_runtime import (
    FlowTargetScheduler,
    capture_relative_scene_state,
    flow_control_ready,
    restore_relative_scene_state,
    settle_after_reset,
    set_flow_control_ready,
)


ROOT = Path(__file__).resolve().parents[1]


def test_replay_buffer_and_balanced_sampling(tmp_path):
    prior = ReplayBuffer(32, observation_dim=7, action_dim=19)
    online = ReplayBuffer(32, observation_dim=7, action_dim=19)
    for buffer, offset in ((prior, 0.0), (online, 10.0)):
        buffer.add(
            torch.full((16, 7), offset),
            torch.zeros(16, 19),
            torch.zeros(16, 1),
            torch.full((16, 7), offset + 1.0),
            torch.zeros(16, 1),
        )
    batch = balanced_replay_sample(
        prior, online, 12, prior_fraction=0.5, rng=np.random.default_rng(0)
    )
    assert batch.observation.shape == (12, 7)
    assert int((batch.observation[:, 0] == 0.0).sum()) == 6
    assert int((batch.observation[:, 0] == 10.0).sum()) == 6

    prior.save(tmp_path / "replay")
    restored = ReplayBuffer(32, observation_dim=7, action_dim=19)
    restored.restore(tmp_path / "replay")
    assert restored.size == prior.size
    assert restored.position == prior.position
    np.testing.assert_array_equal(restored.observation[: restored.size], prior.observation[: prior.size])


def test_replay_rejects_nonfinite_values():
    replay = ReplayBuffer(8, observation_dim=3, action_dim=19)
    with pytest.raises(ValueError, match="NaN"):
        replay.add(
            torch.tensor([[float("nan"), 0.0, 0.0]]),
            torch.zeros(1, 19),
            torch.zeros(1, 1),
            torch.zeros(1, 3),
            torch.zeros(1, 1),
        )


def test_rlpd_update_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(4)
    config = RLPDConfig(
        observation_dim=11,
        hidden_dim=32,
        hidden_layers=2,
        critic_ensemble_size=4,
        target_critic_sample_size=2,
    )
    agent = RLPDAgent(config)
    replay = ReplayBuffer(64, observation_dim=11, action_dim=19)
    replay.add(
        torch.randn(32, 11),
        torch.tanh(torch.randn(32, 19)),
        torch.randn(32, 1),
        torch.randn(32, 11),
        torch.zeros(32, 1),
    )
    before = [value.detach().clone() for value in agent.actor.parameters()]
    metrics = agent.update(replay.sample(16, rng=np.random.default_rng(2)))
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.02 <= metrics["residual_std_mean"] <= 0.25
    assert 0.02 <= metrics["residual_std_max"] <= 0.25
    assert any(not torch.equal(old, new) for old, new in zip(before, agent.actor.parameters()))
    assert agent.act(torch.zeros(3, 11), deterministic=True).shape == (3, 19)

    agent.save_pretrained(tmp_path)
    restored = RLPDAgent.from_pretrained(tmp_path)
    observation = torch.randn(2, 11)
    torch.testing.assert_close(
        agent.act(observation, deterministic=True),
        restored.act(observation, deterministic=True),
    )
    assert restored.update_steps == 1

    training_state = tmp_path / "training_state.pt"
    agent.save_training_state(training_state)
    training_restored = RLPDAgent(config)
    training_restored.load_training_state(training_state)
    assert training_restored.update_steps == agent.update_steps
    for expected, actual in zip(
        agent.critic.parameters(), training_restored.critic.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual)


def test_rlpd_critic_warmup_preserves_actor_and_reports_prior_bc():
    config = RLPDConfig(
        observation_dim=11,
        hidden_dim=32,
        hidden_layers=2,
        critic_ensemble_size=4,
        target_critic_sample_size=2,
    )
    agent = RLPDAgent(config)
    prior = torch.linspace(-0.6, 0.6, config.action_dim)
    agent.initialize_actor_residual(prior)
    before = [value.detach().clone() for value in agent.actor.parameters()]
    replay = ReplayBuffer(64, observation_dim=11, action_dim=19)
    replay.add(
        torch.randn(32, 11),
        prior.expand(32, -1),
        torch.randn(32, 1),
        torch.randn(32, 11),
        torch.zeros(32, 1),
    )

    metrics = agent.update(
        replay.sample(16, rng=np.random.default_rng(2)),
        prior_count=8,
        update_actor=False,
    )
    assert metrics["actor_updated"] == 0.0
    assert metrics["prior_bc_loss"] == pytest.approx(0.0, abs=1.0e-10)
    assert all(torch.equal(old, new) for old, new in zip(before, agent.actor.parameters()))


def test_rlpd_actor_uses_frozen_reference_and_normalized_q_objective():
    config = RLPDConfig(
        observation_dim=11,
        hidden_dim=32,
        hidden_layers=2,
        critic_ensemble_size=4,
        target_critic_sample_size=2,
    )
    agent = RLPDAgent(config)
    reference_before = [value.detach().clone() for value in agent.reference_actor.parameters()]
    replay = ReplayBuffer(64, observation_dim=11, action_dim=19)
    replay.add(
        torch.randn(32, 11),
        torch.zeros(32, 19),
        torch.randn(32, 1),
        torch.randn(32, 11),
        torch.zeros(32, 1),
    )

    metrics = agent.update(replay.sample(16, rng=np.random.default_rng(2)))

    assert metrics["actor_q_multiplier"] > 0.0
    assert metrics["reference_bc_loss"] >= 0.0
    assert metrics["temperature_loss"] == 0.0
    assert all(
        torch.equal(old, new)
        for old, new in zip(reference_before, agent.reference_actor.parameters(), strict=True)
    )


def test_residual_actor_std_is_initialized_and_bounded():
    config = RLPDConfig(
        observation_dim=8,
        hidden_dim=16,
        hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_sample_size=2,
    )
    agent = RLPDAgent(config)
    observation = torch.zeros(3, config.observation_dim)
    _mean, log_std = agent.actor.distribution_parameters(observation)
    torch.testing.assert_close(
        log_std.exp(),
        torch.full_like(log_std, config.initial_residual_std),
    )

    final = agent.actor.network[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias[config.action_dim :].fill_(100.0)
    _mean, log_std = agent.actor.distribution_parameters(observation)
    torch.testing.assert_close(
        log_std.exp(),
        torch.full_like(log_std, config.max_residual_std),
    )
    with torch.no_grad():
        final.bias[config.action_dim :].fill_(-100.0)
    _mean, log_std = agent.actor.distribution_parameters(observation)
    torch.testing.assert_close(
        log_std.exp(),
        torch.full_like(log_std, config.min_residual_std),
    )


def test_residual_actor_rejects_degenerate_std_interval():
    with pytest.raises(ValueError, match="min < max"):
        RLPDConfig(
            observation_dim=8,
            min_residual_std=0.005,
            initial_residual_std=0.005,
            max_residual_std=0.005,
        )


def test_stage_mask_preserves_right_first_lift_before_left_hand_join():
    mask = stochastic_action_mask_for_stage("contact")
    assert [index for index, enabled in enumerate(mask) if enabled] == [
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
    ]
    lift_mask = stochastic_action_mask_for_stage("sequential_lift")
    assert [index for index, enabled in enumerate(lift_mask) if enabled] == [
        0,
        1,
        2,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
    ]
    # The next curriculum stage must be able to move both arms and both Dex1
    # commands so the left hand can join the right-lifted table.
    assert stochastic_action_mask_for_stage("lift") == (1.0,) * 19
    assert stochastic_action_mask_for_stage("flip") == (1.0,) * 19
    with pytest.raises(ValueError, match="unknown flip-table curriculum stage"):
        stochastic_action_mask_for_stage("invalid")


def test_target_safety_factory_reads_deployable_runtime_limits(monkeypatch):
    monkeypatch.setenv("FLIP_TABLE_RLPD_BODY_TARGET_VELOCITY_LIMIT_RAD_S", "1.5")
    monkeypatch.setenv("FLIP_TABLE_RLPD_BODY_TARGET_ACCELERATION_LIMIT_RAD_S2", "1.25")
    safety = target_safety_from_environment(policy_hz=30.0)

    assert safety.policy_hz == 30.0
    assert safety.body_velocity_limit_rad_s == 1.5
    assert safety.body_acceleration_limit_rad_s2 == 1.25


def test_residual_actor_mask_makes_inactive_axes_deterministic():
    mask = stochastic_action_mask_for_stage("contact")
    config = RLPDConfig(
        observation_dim=8,
        hidden_dim=16,
        hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_sample_size=2,
        stochastic_action_mask=mask,
    )
    agent = RLPDAgent(config)
    observation = torch.randn(64, config.observation_dim)
    deterministic = agent.act(observation, deterministic=True)
    stochastic = agent.act(observation)
    inactive = torch.tensor(mask) == 0
    active = ~inactive
    torch.testing.assert_close(
        stochastic[:, inactive],
        torch.zeros_like(stochastic[:, inactive]),
    )
    torch.testing.assert_close(
        deterministic[:, inactive],
        torch.zeros_like(deterministic[:, inactive]),
    )
    assert not torch.equal(stochastic[:, active], deterministic[:, active])


def test_rlpd_config_rejects_invalid_stochastic_action_masks():
    with pytest.raises(ValueError, match="exactly 19"):
        RLPDConfig(observation_dim=8, stochastic_action_mask=(1.0,) * 18)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        RLPDConfig(observation_dim=8, stochastic_action_mask=(0.5,) + (1.0,) * 18)
    with pytest.raises(ValueError, match="enable at least one"):
        RLPDConfig(observation_dim=8, stochastic_action_mask=(0.0,) * 19)


def test_rlpd_actor_can_initialize_from_a_successful_residual_prior():
    config = RLPDConfig(
        observation_dim=8,
        hidden_dim=16,
        hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_sample_size=2,
    )
    agent = RLPDAgent(config)
    prior = torch.linspace(-0.7, 0.7, config.action_dim)
    agent.initialize_actor_residual(prior)

    action = agent.act(torch.randn(4, config.observation_dim), deterministic=True)
    torch.testing.assert_close(action, prior.expand_as(action), atol=1.0e-6, rtol=0.0)

    with pytest.raises(ValueError, match="strictly inside"):
        agent.initialize_actor_residual(torch.ones(config.action_dim))


def test_rlpd_config_accepts_legacy_unit_residual_scale_only():
    payload = RLPDConfig(observation_dim=8).to_dict()
    payload["residual_scale_rad"] = 1.0
    restored = RLPDConfig.from_dict(payload)
    assert restored.observation_dim == 8

    payload["residual_scale_rad"] = 2.0
    with pytest.raises(ValueError, match="legacy residual_scale_rad"):
        RLPDConfig.from_dict(payload)


def test_rlpd_cuda_rng_restore_uses_cpu_byte_tensors(tmp_path, monkeypatch):
    config = RLPDConfig(
        observation_dim=8,
        hidden_dim=16,
        hidden_layers=1,
        critic_ensemble_size=2,
        target_critic_sample_size=2,
    )
    agent = RLPDAgent(config)
    checkpoint = tmp_path / "training_state.pt"
    agent.save_training_state(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["cuda_rng_states"] = [torch.arange(16, dtype=torch.uint8)]
    torch.save(payload, checkpoint)

    restored = RLPDAgent(config)
    restored.device = torch.device("cuda")
    captured = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: captured.append(states))
    restored.load_training_state(checkpoint)

    assert len(captured) == 1
    assert all(state.device.type == "cpu" for state in captured[0])
    assert all(state.dtype == torch.uint8 for state in captured[0])


def test_residual_contract_uses_radians_for_body_and_command_scale_for_hands():
    base = torch.zeros(1, 19)
    base[:, 17:] = torch.tensor([0.0, 4.5])
    residual = torch.ones(1, 19)
    result = apply_residual_to_base(base, residual)

    torch.testing.assert_close(result[0, :17], torch.tensor(BODY_RESIDUAL_SCALE))
    assert result[0, 17].item() == pytest.approx(0.0)
    assert result[0, 18].item() == pytest.approx(0.0)

    reverse = apply_residual_to_base(base, -residual)
    assert reverse[0, 17].item() == pytest.approx(4.5)
    assert reverse[0, 18].item() == pytest.approx(4.5)


def test_policy_clock_maps_30_hz_to_50_hz_without_drift():
    clock = PolicyControlClock(policy_hz=30.0, sim_control_hz=50.0)
    boundaries = [clock.advance_sim_step() for _ in range(50)]

    assert sum(boundaries) == 30
    assert boundaries[:10] == [False, True, False, True, True, False, True, False, True, True]
    assert clock.phase == pytest.approx(0.0)
    assert clock.completed_policy_intervals == 30

    clock.reset()
    assert clock.sim_steps == 0
    assert clock.completed_policy_intervals == 0
    assert clock.phase == 0.0


def test_policy_clock_rejects_unrepresentable_faster_policy():
    with pytest.raises(ValueError, match="cannot exceed"):
        PolicyControlClock(policy_hz=60.0, sim_control_hz=50.0)


def test_target_safety_filter_matches_deploy_time_slew_limits():
    safety = UpperBodyTargetSafetyFilter(policy_hz=30.0)
    current = torch.zeros(1, 19)
    target = torch.full((1, 19), 10.0)

    safe, clipped = safety.filter(target, current)

    expected_first_step = (75.0 / 30.0) / 30.0
    torch.testing.assert_close(
        safe[0, :17], torch.full((17,), expected_first_step), rtol=1.0e-5, atol=1.0e-6
    )
    expected_first_hand_step = (400.0 / 30.0) / 30.0
    torch.testing.assert_close(
        safe[0, 17:], torch.full((2,), expected_first_hand_step)
    )
    assert clipped == 19

    safety.reset()
    assert safety.previous_target is None
    assert safety.previous_velocity is None
    assert safety.previous_hand_target is None
    assert safety.previous_hand_velocity is None


def test_target_safety_filter_rejects_nonfinite_measured_state():
    safety = UpperBodyTargetSafetyFilter()
    state = torch.zeros(1, 19)
    state[0, 0] = float("nan")
    with pytest.raises(ValueError, match="current_state"):
        safety.filter(torch.zeros_like(state), state)


def test_target_safety_filter_preserves_contact_loaded_command_target():
    safety = UpperBodyTargetSafetyFilter(policy_hz=30.0)
    commanded = torch.zeros(1, 19)
    commanded[:, 10:17] = torch.tensor(
        [-0.9, -0.6, 0.5, 1.3, -1.0, -0.4, 0.25]
    )
    commanded[:, 18] = 0.0
    measured = commanded.clone()
    # A position-controlled gripper under load cannot reach its closed target.
    measured[:, 18] = 0.15
    safety.reset(commanded)

    safe, clipped = safety.filter(commanded, measured)

    torch.testing.assert_close(safe, commanded)
    assert clipped == 0


def test_absolute_target_delay_starts_from_measured_state():
    delay = AbsoluteTargetDelayBuffer(num_envs=1, max_delay_steps=1, device="cpu")
    state = torch.full((1, 19), 0.25)
    delay.reset(state)
    delay.delay_steps.fill_(1)
    first_target = torch.full((1, 19), 0.75)
    second_target = torch.full((1, 19), 1.25)

    torch.testing.assert_close(delay.apply(first_target), state)
    torch.testing.assert_close(delay.apply(second_target), first_target)


def test_absolute_target_delay_requires_reset():
    delay = AbsoluteTargetDelayBuffer(num_envs=1, max_delay_steps=0, device="cpu")
    with pytest.raises(RuntimeError, match="reset"):
        delay.apply(torch.zeros(1, 19))


def test_flow_target_scheduler_uses_the_checkpoint_action_prefix():
    class FakeFlow:
        config = SimpleNamespace(n_action_steps=2)

        def __init__(self) -> None:
            self.calls = 0

        def sample_actions(self, images, state):
            self.calls += 1
            chunk = torch.arange(3 * 19, dtype=torch.float32).reshape(1, 3, 19)
            return chunk + 100.0 * (self.calls - 1)

    flow = FakeFlow()
    scheduler = FlowTargetScheduler(flow)
    images = torch.zeros(1, 3, 3, 4, 4)
    state = torch.zeros(1, 19)

    torch.testing.assert_close(scheduler.current(images, state), torch.arange(19.0).reshape(1, 19))
    scheduler.advance()
    torch.testing.assert_close(
        scheduler.current(images, state),
        torch.arange(19.0, 38.0).reshape(1, 19),
    )
    scheduler.advance()
    torch.testing.assert_close(
        scheduler.current(images, state),
        torch.arange(100.0, 119.0).reshape(1, 19),
    )
    assert flow.calls == 2

    scheduler.reset()
    assert scheduler.chunk is None
    assert scheduler.index == 0


def test_flow_target_scheduler_reanchors_with_deployable_joint_state():
    class FakeFlow:
        config = SimpleNamespace(n_action_steps=2)

        def sample_actions(self, images, state):
            return torch.stack((state + 0.4, state + 0.5), dim=1)

    scheduler = FlowTargetScheduler(FakeFlow())
    images = torch.zeros(1, 3, 3, 4, 4)
    state = torch.arange(19, dtype=torch.float32).reshape(1, 19)

    torch.testing.assert_close(scheduler.anchor_to_state(images, state), state)
    scheduler.advance()
    torch.testing.assert_close(scheduler.current(images, state), state + 0.1)

    scheduler.reset()
    assert scheduler.anchor_offset is None

    hold_scheduler = FlowTargetScheduler(FakeFlow(), motion_gain=0.0)
    torch.testing.assert_close(hold_scheduler.anchor_to_state(images, state), state)
    hold_scheduler.advance()
    torch.testing.assert_close(hold_scheduler.current(images, state), state)

    body_only = FlowTargetScheduler(
        FakeFlow(),
        motion_mask=tuple([1.0] * 17 + [0.0, 0.0]),
    )
    torch.testing.assert_close(body_only.anchor_to_state(images, state), state)
    body_only.advance()
    expected = state + 0.1
    expected[:, 17:] = state[:, 17:]
    torch.testing.assert_close(body_only.current(images, state), expected)


def test_flow_target_scheduler_rejects_invalid_motion_mask():
    flow = SimpleNamespace(config=SimpleNamespace(n_action_steps=1))
    with pytest.raises(ValueError, match="19 binary"):
        FlowTargetScheduler(flow, motion_mask=(1.0,) * 18)


def test_flow_control_handoff_readiness_is_explicit():
    env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    torch.testing.assert_close(flow_control_ready(env), torch.ones(2, dtype=torch.bool))

    set_flow_control_ready(env, False)
    torch.testing.assert_close(flow_control_ready(env), torch.zeros(2, dtype=torch.bool))
    set_flow_control_ready(env, True)
    torch.testing.assert_close(flow_control_ready(env), torch.ones(2, dtype=torch.bool))


def test_reset_settle_steps_hold_state_and_discard_sensor_ticks():
    class FakeEnvironment:
        num_envs = 2

    class FakeGymEnvironment:
        def __init__(self, env):
            self.env = env
            self.actions = []
            self.targets = []

        def step(self, action):
            self.actions.append(action.clone())
            self.targets.append(self.env._flip_table_rlpd_absolute_target.clone())
            done = torch.zeros(2, dtype=torch.bool)
            return None, torch.zeros(2), done, done, {}

    env = FakeEnvironment()
    gym_env = FakeGymEnvironment(env)
    state = torch.arange(38, dtype=torch.float32).reshape(2, 19)
    settled_steps = []

    settled = settle_after_reset(
        gym_env,
        env,
        steps=2,
        state_reader=lambda _env: state,
        post_step=lambda _env, step: settled_steps.append(step),
    )

    assert settled == 2
    assert settled_steps == [0, 1]
    assert len(gym_env.actions) == 2
    for action, target in zip(gym_env.actions, gym_env.targets, strict=True):
        torch.testing.assert_close(action, torch.zeros_like(state))
        torch.testing.assert_close(target, state)

    with pytest.raises(ValueError, match="cannot be negative"):
        settle_after_reset(gym_env, env, steps=-1, state_reader=lambda _env: state)


def test_relative_scene_state_capture_and_restore_are_isolated():
    source_pose = torch.tensor([[1.0, 2.0, 3.0]])

    class FakeScene:
        def get_state(self, *, is_relative):
            assert is_relative is True
            return {"rigid_object": {"table": {"root_pose": source_pose}}}

    class FakeEnvironment:
        num_envs = 1
        scene = FakeScene()
        episode_length_buf = torch.zeros(1, dtype=torch.long)

        def __init__(self):
            self.restored = None

        def reset_to(self, state, *, env_ids, is_relative):
            self.restored = (state, env_ids, is_relative)

    env = FakeEnvironment()
    state = capture_relative_scene_state(env)
    source_pose.fill_(99.0)
    torch.testing.assert_close(
        state["rigid_object"]["table"]["root_pose"],
        torch.tensor([[1.0, 2.0, 3.0]]),
    )

    restore_relative_scene_state(env, state, episode_step=1250)

    restored_state, env_ids, is_relative = env.restored
    assert restored_state is state
    assert env_ids is None
    assert is_relative is True
    torch.testing.assert_close(env.episode_length_buf, torch.tensor([1250]))
    with pytest.raises(ValueError, match="cannot be negative"):
        restore_relative_scene_state(env, state, episode_step=-1)


def test_rlpd_runner_enforces_synchronized_hard_safety_resets():
    source = (ROOT / "scripts" / "train_flow_residual_rlpd.py").read_text(encoding="utf-8")

    assert "FLIP_TABLE_RLPD_HARD_RESET_FINGER_FORCE_N" in source
    assert 'FLIP_TABLE_RLPD_HARD_RESET_FINGER_FORCE_N", "15.1"' in source
    assert "hard_reset_finger_force_n <= mdp.MAX_SAFE_FINGER_FORCE_N" in source
    assert '"env_spacing_m": float(env.cfg.scene.env_spacing)' in source
    assert "per-environment maxima" in source
    assert "episode_ever_success & episode_safe" in source
    assert "synchronized_stage_success = bool(stage_success.any())" in source
    assert "manual_reset = (hard_reset or synchronized_stage_success)" in source
    assert "done_bool = torch.ones_like(done_bool, dtype=torch.bool)" in source
    assert "gym_env.reset()" in source
    assert "settle_after_reset(" in source
    assert "restore_deployable_prefix_state(" in source
    assert "capture_relative_scene_state(env)" in source
    assert "prefix-state reuse is forbidden when domain randomization is active" in source
    assert '"final_evaluation_must_execute_full_prefix": True' in source
    assert "agent.initialize_actor_residual(prior_residual[0])" in source
    assert "min_residual_std=min(0.001, 0.5 * args_cli.random_residual_std)" in source
    assert "initial_residual_std=args_cli.random_residual_std" in source
    assert "max_residual_std=args_cli.random_residual_std" in source
    assert "0.001 <= args_cli.random_residual_std <= 0.25" in source
    assert "traceback.print_exc()" in source
    assert "critic_warmup_updates" in source
    assert "prior_count=args_cli.batch_size // 2" in source
    assert "update_actor=update_actor" in source
    assert "residual = prior_residual.expand_as(previous_residual).clone()" in source
    assert "immutable prior residual was modified during training" in source
    assert "exploration_center\n                    + torch.randn_like" in source
    assert "* stochastic_action_mask_tensor" in source
    assert '"prior_residual"' in source
    assert "source_agent = RLPDAgent.from_pretrained" in source
    assert "agent.actor.load_state_dict(source_agent.actor.state_dict(), strict=True)" in source
    assert "agent.set_reference_actor_from_current()" in source
    assert 'args_cli.prior_action_source == "actor"' in source
    assert "agent.act(features, deterministic=True).clone()" in source
    assert '"training_prior_action_source"' in source
    assert "reference_bc_weight=args_cli.reference_bc_weight" in source
    assert "actor_q_normalization=args_cli.actor_q_normalization" in source
    assert "automatic_entropy_tuning=False" in source


def test_deterministic_rlpd_stage_evaluator_preserves_deployment_contract():
    source = (ROOT / "scripts" / "evaluate_flow_residual_rlpd.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "run_train_in_container.sh").read_text(encoding="utf-8")

    assert "agent.act(features, deterministic=True)" in source
    assert 'choices=("policy", "zero", "constant", "policy_plus_constant")' in source
    assert '"--flow-checkpoint"' in source
    assert "provide exactly one of --checkpoint or --flow-checkpoint" in source
    assert 'args_cli.residual_mode not in {"zero", "constant"}' in source
    assert 'checkpoint_kind = "standalone_flow_matching"' in source
    assert "assert agent is not None" in source
    assert 'args_cli.residual_mode == "zero"' in source
    assert 'args_cli.residual_mode == "constant"' in source
    assert 'args_cli.residual_mode == "policy_plus_constant"' in source
    assert 'args_cli.residual_mode in {"constant", "policy_plus_constant"}' in source
    assert "(residual + constant_residual).clamp(-1.0, 1.0)" in source
    assert "_parse_constant_residual" in source
    assert "args_cli.max_sim_steps / args_cli.sim_control_hz" in source
    assert "explicit_cli" in source
    assert "for name, value in explicit_cli.items()" in source
    assert "residual_observation(" in source
    assert '"actor_critic_privileged_inputs": []' in source
    assert '"privileged_use": "success, safety and trace diagnostics only"' in source
    assert 'env._flip_table_rlpd_absolute_target = delayed_target' in source
    assert "PolicyControlClock(args_cli.policy_hz, args_cli.sim_control_hz)" in source
    assert "camera_batch(env)" in source
    assert "dataset_joint_state(env)" in source
    assert "settle_after_reset(" in source
    assert "if output.exists():\n        raise FileExistsError(output)" in source
    assert source.index("combined_manifest = _validate_checkpoint_contract") < source.index(
        "output.mkdir(parents=True, exist_ok=False)"
    )
    assert "_parse_episode_seeds" in source
    assert "for episode, episode_seed in enumerate(episode_seeds)" in source
    assert "set_seed(episode_seed, env)" in source
    assert '"explicit_episode_seed_list"' in source
    assert '"base_seed_plus_episode_index"' in source
    assert '"episode_seeds": episode_seeds' in source
    assert "mdp.table_stage_success(env)" in source
    assert "mdp.table_stable_success(env)" in source
    assert '"success": task_success' in source
    assert '"task_success": task_success' in source
    assert '"curriculum_stage_success": curriculum_stage_success' in source
    assert '"curriculum_stage_success_is_task_success": False' in source
    assert "if task_success:" in source
    assert "and args_cli.stop_on_curriculum_stage_success" in source
    assert "--stop-on-curriculum-stage-success" in source
    assert "evaluate_rlpd_stage" in runner
    assert "FLIP_TABLE_RLPD_COMBINED_CHECKPOINT" in runner
    assert "FLIP_TABLE_FLOW_CHECKPOINT" in runner
    assert 'checkpoint_args=(--flow-checkpoint "$FLIP_TABLE_FLOW_CHECKPOINT")' in runner
    assert "FLIP_TABLE_RLPD_RECORD_VIDEO" in runner
    assert "FLIP_TABLE_RLPD_EVAL_RESIDUAL_MODE" in runner
    assert "FLIP_TABLE_RLPD_EVAL_CONSTANT_RESIDUAL" in runner
    assert "FLIP_TABLE_RLPD_EVAL_EPISODE_SEEDS" in runner
    assert "FLIP_TABLE_RLPD_STOP_ON_CURRICULUM_STAGE_SUCCESS" in runner
    assert "FLIP_TABLE_RLPD_HARD_RESET_FINGER_FORCE_N:-15.1" in runner
    assert "FLIP_TABLE_RLPD_RESET_SETTLE_STEPS:-4" in runner
    assert "FLIP_TABLE_RLPD_PRIOR_ACTION_SOURCE" in runner
    assert "FLIP_TABLE_RLPD_ACTOR_INIT_CHECKPOINT" in runner
    assert "FLIP_TABLE_RLPD_REUSE_PREFIX_STATE" in runner
    assert "args+=(--reuse-prefix-state)" in runner
    assert 'export FLIP_TABLE_RL_ACTION_DELAY_MAX_STEPS=0' in runner
