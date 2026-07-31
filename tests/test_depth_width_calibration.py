from __future__ import annotations

import torch

from nvs.depth_width_calibration import (
    COUNTERFACTUAL_MODES,
    build_calibration_comparisons,
)
from nvs.depth_width_observability import _rank_correlation


def _predicted() -> torch.Tensor:
    centers = torch.linspace(-0.4, 0.3, 8)
    widths = torch.tensor([0.1, 0.2, 0.4, 0.8, 1.1, 1.3, 1.7, 2.1])
    return torch.stack((centers, widths), dim=-1).unsqueeze(0)


def test_comparisons_are_deterministic_camera_local_and_center_preserving() -> None:
    predicted = _predicted()
    first = build_calibration_comparisons(
        predicted, num_patches=4, scene_id="scene-a", layer_index=2
    )
    second = build_calibration_comparisons(
        predicted, num_patches=4, scene_id="scene-a", layer_index=2
    )
    references, counterfactuals, _ = first

    assert tuple(counterfactuals) == COUNTERFACTUAL_MODES
    assert len(counterfactuals) == 35
    for name in COUNTERFACTUAL_MODES:
        torch.testing.assert_close(counterfactuals[name], second[1][name])
        torch.testing.assert_close(counterfactuals[name][..., 0], predicted[..., 0])
        torch.testing.assert_close(references[name][..., 0], predicted[..., 0])
    torch.testing.assert_close(
        counterfactuals["dose_s1_cap_inf"], predicted
    )
    assert counterfactuals["dose_s0_cap_inf"][..., 1].count_nonzero() == 0


def test_routing_preserves_each_camera_multiset_and_reverse_rank() -> None:
    predicted = _predicted()
    references, counterfactuals, _ = build_calibration_comparisons(
        predicted, num_patches=4, scene_id="scene-a", layer_index=1
    )

    for routing in ("roll", "random", "reverse"):
        name = f"route_raw_{routing}"
        reference = references[name][0, :, 1].reshape(2, 4)
        changed = counterfactuals[name][0, :, 1].reshape(2, 4)
        for camera in range(2):
            torch.testing.assert_close(
                changed[camera].sort().values,
                reference[camera].sort().values,
            )
    reverse = counterfactuals["route_raw_reverse"][0, :4, 1]
    assert _rank_correlation(predicted[0, :4, 1], reverse) == -1.0


def test_registered_sentinels_apply_the_expected_scale_and_cap() -> None:
    predicted = _predicted()
    references, counterfactuals, _ = build_calibration_comparisons(
        predicted, num_patches=4, scene_id="scene-a", layer_index=0
    )

    low = references["route_low_reverse"][..., 1]
    mid = references["route_mid_reverse"][..., 1]
    torch.testing.assert_close(low, (predicted[..., 1].abs() * 0.125).clamp(max=0.5))
    torch.testing.assert_close(mid, (predicted[..., 1].abs() * 0.25).clamp(max=1.0))
