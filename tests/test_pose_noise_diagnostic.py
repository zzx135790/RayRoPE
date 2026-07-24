"""Tests for clean-training pose-noise diagnostics."""

from types import SimpleNamespace

from nvs.trainval import LVSMLauncher, _model_uses_pose_sigma


def test_blind_pose_noise_diagnostic_is_separate_from_noisy_training():
    launcher = object.__new__(LVSMLauncher)
    launcher.config = SimpleNamespace(
        pose_noise_enabled=False,
        pose_noise_diagnostic_only=True,
    )
    calls = []
    launcher.pose_noise_test_sweep = lambda step, state: calls.append((step, state))
    state = {"model": object()}

    launcher.run_pose_noise_diagnostic(step=49, state=state)

    assert calls == [(49, state)]


def test_pose_noise_diagnostic_is_disabled_by_default():
    launcher = object.__new__(LVSMLauncher)
    launcher.config = SimpleNamespace(pose_noise_diagnostic_only=False)
    calls = []
    launcher.pose_noise_test_sweep = lambda step, state: calls.append((step, state))

    launcher.run_pose_noise_diagnostic(step=49, state={})

    assert calls == []


def test_pose_sigma_routing_includes_stochastic_and_cf_modes():
    assert _model_uses_pose_sigma(
        SimpleNamespace(use_pose_uncertainty=True, use_pose_uncertainty_cf=False)
    )
    assert _model_uses_pose_sigma(
        SimpleNamespace(use_pose_uncertainty=False, use_pose_uncertainty_cf=True)
    )
    assert not _model_uses_pose_sigma(
        SimpleNamespace(use_pose_uncertainty=False, use_pose_uncertainty_cf=False)
    )
