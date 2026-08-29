"""Tests for clean-training pose-noise diagnostics."""

from types import SimpleNamespace

import torch
import pytest

from nvs.lvsm import _transform_depth_uncertainty
from nvs.trainval import (
    LVSMLauncher,
    _camera_input_digest,
    _curriculum_noise_values,
    _exact_resume_data_alignment,
    _effective_uncertainty_mc_samples,
    _model_uses_pose_sigma,
    _pose_level_is_noisy,
    _pose_noise_level_seed,
    _pose_noise_train_seed,
    _uncertainty_forward_seed,
    _uncertainty_intervention_modes,
)


def test_default_coupled_axis_preserves_historical_metric_tags():
    launcher = object.__new__(LVSMLauncher)
    launcher.config = SimpleNamespace(
        pose_noise_test_axes="coupled",
        pose_noise_test_levels="0,0.01,0.02",
    )

    assert launcher._pose_test_levels() == [
        (0.0, 0.0, "clean", "clean"),
        (0.01, 0.01, "rot0.01", "coupled"),
        (0.02, 0.02, "rot0.02", "coupled"),
    ]


def test_explicit_multi_axis_panel_keeps_axes_separate_and_translation_is_noisy():
    launcher = object.__new__(LVSMLauncher)
    launcher.config = SimpleNamespace(
        pose_noise_test_axes="rotation_only,translation_only,coupled",
        pose_noise_test_levels="0,0.01",
    )

    assert launcher._pose_test_levels() == [
        (0.0, 0.0, "clean", "clean"),
        (0.01, 0.0, "rotation_only-0.01", "rotation_only"),
        (0.0, 0.01, "translation_only-0.01", "translation_only"),
        (0.01, 0.01, "coupled-0.01", "coupled"),
    ]
    assert _pose_level_is_noisy(0.0, 0.01)
    assert not _pose_level_is_noisy(0.0, 0.0)
    assert _pose_noise_level_seed(1234, 0.0, 0.01, 7) == (
        _pose_noise_level_seed(1234, 0.01, 0.0, 7)
    )


def test_exact_resume_accepts_only_a_dataloader_epoch_boundary():
    assert _exact_resume_data_alignment(data_cursor=10_000, batches_per_epoch=16) == 0
    with pytest.raises(ValueError, match="epoch boundary"):
        _exact_resume_data_alignment(data_cursor=9_999, batches_per_epoch=16)


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
    assert _model_uses_pose_sigma(
        SimpleNamespace(
            use_pose_uncertainty=False,
            use_pose_uncertainty_cf=False,
            uncertainty_strategy="linearized_shared_sample",
        )
    )


def test_corruption_digest_tracks_exact_camera_inputs():
    first = SimpleNamespace(camtoworld=torch.eye(4).reshape(1, 1, 4, 4))
    same = SimpleNamespace(camtoworld=first.camtoworld.clone())
    changed = SimpleNamespace(camtoworld=first.camtoworld.clone())
    changed.camtoworld[0, 0, 0, 3] = 0.01

    assert _camera_input_digest(first) == _camera_input_digest(same)
    assert _camera_input_digest(first) != _camera_input_digest(changed)


def test_double_linear_curriculum_has_locked_endpoints_and_midpoint():
    assert _curriculum_noise_values(0, 100, 0.5, 0.03, 0.03) == (0.0, 0.0, 0.0)
    assert _curriculum_noise_values(50, 100, 0.5, 0.03, 0.03) == pytest.approx(
        (0.25, 0.015, 0.015)
    )
    assert _curriculum_noise_values(100, 100, 0.5, 0.03, 0.03) == pytest.approx(
        (0.5, 0.03, 0.03)
    )
    assert _curriculum_noise_values(200, 100, 0.5, 0.03, 0.03) == pytest.approx(
        (0.5, 0.03, 0.03)
    )


def test_owner_forward_seed_is_stable_bounded_and_namespaced():
    first = _uncertainty_forward_seed(2, 14_999, 1, 3)
    assert first == _uncertainty_forward_seed(2, 14_999, 1, 3)
    assert 0 <= first < 2**63
    assert len({
        _uncertainty_forward_seed(2, 14_999, 1, 3),
        _uncertainty_forward_seed(1, 14_999, 1, 3),
        _uncertainty_forward_seed(2, 14_998, 1, 3),
        _uncertainty_forward_seed(2, 14_999, 0, 3),
        _uncertainty_forward_seed(2, 14_999, 1, 2),
    }) == 5


def test_pose_noise_seed_changes_for_each_gradient_accumulation_microbatch():
    first = _pose_noise_train_seed(1234, 41, 0)

    assert first == _pose_noise_train_seed(1234, 41, 0)
    assert 0 <= first < 2**63
    assert len({
        _pose_noise_train_seed(1234, 41, 0),
        _pose_noise_train_seed(1234, 41, 1),
        _pose_noise_train_seed(1234, 42, 0),
        _pose_noise_train_seed(1235, 41, 0),
    }) == 4


@pytest.mark.parametrize(
    ("intervention", "expected"),
    [
        ("true", ("true", "true")),
        ("pose_zero", ("zero", "true")),
        ("pose_permute", ("permute", "true")),
        ("depth_zero", ("true", "zero")),
        ("depth_permute_within_camera", ("true", "permute_within_camera")),
        ("all_zero", ("zero", "zero")),
    ],
)
def test_uncertainty_intervention_modes_are_independent(intervention, expected):
    assert _uncertainty_intervention_modes(intervention) == expected


def test_none_uses_one_effective_mc_sample_but_sampled_modes_keep_request():
    assert _effective_uncertainty_mc_samples(
        SimpleNamespace(uncertainty_strategy="none"), 4
    ) == 1
    assert _effective_uncertainty_mc_samples(
        SimpleNamespace(uncertainty_strategy="linearized_shared_sample"), 4
    ) == 4


def test_depth_zero_preserves_centres_and_removes_widths():
    predicted = torch.tensor([[[1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [4.0, 0.4]]])
    transformed = _transform_depth_uncertainty(predicted, "zero", num_patches=2)

    torch.testing.assert_close(transformed[..., 0], predicted[..., 0])
    torch.testing.assert_close(transformed[..., 1], torch.zeros_like(predicted[..., 1]))
    torch.testing.assert_close(predicted[..., 1], torch.tensor([[0.1, 0.2, 0.3, 0.4]]))


def test_depth_permutation_stays_within_each_camera_and_preserves_centres():
    predicted = torch.tensor([[[1.0, 0.1], [2.0, 0.2], [3.0, 1.1], [4.0, 1.2]]])
    transformed = _transform_depth_uncertainty(
        predicted, "permute_within_camera", num_patches=2
    )

    torch.testing.assert_close(transformed[..., 0], predicted[..., 0])
    torch.testing.assert_close(
        transformed[..., 1], torch.tensor([[0.2, 0.1, 1.2, 1.1]])
    )
