import pytest
import torch
import torch.nn.functional as F

from nvs.lvsm import (
    LVSMDecoderOnlyModel,
    LVSMDecoderOnlyModelConfig,
    _prepare_depth_input,
    _sample_layerwise_depth_tokens,
)
from pos_enc.utils.functional import Camera
from pos_enc.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)


def _config(
    mode: str = "joint",
    *,
    policy: str | None = None,
    num_layers: int = 3,
    family: str = "ray_point",
    depth_input: bool = False,
    amplitude_intervention: str = "true",
    edge_mask: tuple[bool, bool, bool, bool, bool] = (True,) * 5,
    legacy_compatibility: bool = False,
    trace_enabled: bool = True,
    update_parameterization: str = "bounded_residual",
):
    return LVSMDecoderOnlyModelConfig(
        ref_views=1,
        tar_views=1,
        encoder=TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=192,
                nhead=4,
                dim_feedforward=384,
                dropout=0.0,
                activation=F.relu,
                layer_norm_eps=1e-5,
                batch_first=True,
                norm_first=True,
                bias=False,
                elementwise_affine=True,
                norm_type="layer_norm",
                modulation_activation=None,
                qk_norm=False,
            ),
            num_layers=num_layers,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
        img_shape=(16, 16, 3),
        cam_shape=(16, 16, 6),
        patch_size=8,
        pos_enc="flag_rope",
        depth_type="none",
        depth_input=depth_input,
        rope_family=family,
        layerwise_belief_mode=mode,
        layerwise_belief_policy=policy,
        layerwise_update_parameterization=update_parameterization,
        layerwise_legacy_compatibility=legacy_compatibility,
        layerwise_edge_mask=edge_mask,
        layerwise_amplitude_intervention=amplitude_intervention,
        layerwise_trace_enabled=trace_enabled,
        scene_scale_value=1.0,
        freq_min_lambda=0.1,
        freq_max_lambda=2.0,
    )


def test_non_geometry_attention_does_not_require_geometry_precompute() -> None:
    config = _config("none", num_layers=2)
    config.pos_enc = "none"
    model = LVSMDecoderOnlyModel(config)
    output = model(
        torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras()
    )
    assert output.shape == (1, 1, 16, 16, 3)


def _cameras(batch: int = 1):
    K = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, 1, 1, 1)
    pose = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, 1, 1, 1)
    return Camera(K=K, camtoworld=pose, width=16, height=16)


def test_external_depth_sampler_preserves_holes_and_adds_unavailable_target() -> None:
    depth = torch.ones(1, 1, 16, 16, 1)
    depth[:, :, :8, :8] = 2.0
    depth[:, :, :8, 8:] = 0.0
    depth[:, :, 8:, :8] = float("nan")
    sampled, available = _sample_layerwise_depth_tokens(
        depth, target_views=1, patch_size=8, reference_views=1
    )
    assert sampled is not None and available is not None
    assert sampled.shape == (1, 8)
    assert available.shape == (1, 8)
    assert torch.equal(available[0, :4], torch.tensor([True, False, False, True]))
    assert torch.equal(available[0, 4:], torch.zeros(4, dtype=torch.bool))
    assert sampled[0, 0] == 2.0
    assert sampled[0, 1] == 1.0
    assert sampled[0, 4:].eq(1.0).all()


def test_depth_input_converts_holes_to_finite_sentinel() -> None:
    images = torch.zeros(1, 1, 16, 16, 3)
    depths = torch.ones(1, 1, 16, 16, 1)
    depths[:, :, :8, 8:] = 0.0
    depths[:, :, 8:, :] = float("nan")
    inverse = _prepare_depth_input(depths, reference_images=images)
    assert torch.isfinite(inverse).all()
    assert inverse[0, 0, 0, 0, 0] == 1.0
    assert inverse[0, 0, 0, 8, 0] == 0.0
    assert inverse[0, 0, 8:, :, :].eq(0.0).all()


def test_layerwise_forward_with_depth_input_gracefully_handles_holes() -> None:
    torch.manual_seed(11)
    model = LVSMDecoderOnlyModel(
        _config("joint", num_layers=2, depth_input=True)
    )
    depths = torch.ones(1, 1, 16, 16, 1)
    depths[:, :, :8, 8:] = 0.0
    depths[:, :, 8:, :] = float("nan")
    output = model(
        torch.randn(1, 1, 16, 16, 3),
        _cameras(),
        _cameras(),
        context_depths=depths,
    )
    assert torch.isfinite(output).all()
    assert model.last_layerwise_trace[0]["available_depth_tokens"] == 1


def test_layerwise_lvsm_has_one_layer_lag_and_no_final_side_head() -> None:
    torch.manual_seed(3)
    model = LVSMDecoderOnlyModel(_config("joint", num_layers=3))
    ref_cams = _cameras()
    tar_cams = _cameras()
    imgs = torch.randn(1, 1, 16, 16, 3)
    output = model(imgs, ref_cams, tar_cams)
    assert output.shape == (1, 1, 16, 16, 3)
    trace = model.last_layerwise_trace
    assert [row["layer_index"] for row in trace] == [0, 1, 2]
    assert [row["uncertainty_active"] for row in trace] == [False, True, True]
    assert trace[0]["available_depth_tokens"] == 0
    assert trace[1]["available_depth_tokens"] == 0
    assert len(model.layerwise_belief_refiner.side_heads) == 2


def test_recurrent_delta_is_opt_in_and_threads_into_runtime_trace() -> None:
    torch.manual_seed(13)
    model = LVSMDecoderOnlyModel(
        _config(
            "joint",
            policy="joint",
            num_layers=3,
            update_parameterization="recurrent_delta",
        )
    )
    output = model(
        torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras()
    )
    assert output.shape == (1, 1, 16, 16, 3)
    assert (
        model.layerwise_belief_refiner.config.update_parameterization
        == "recurrent_delta"
    )
    trace = model.last_layerwise_trace
    assert trace[0]["update_parameterization"] == "recurrent_delta"
    assert trace[0]["relative_pose_max_abs"] == 0
    assert trace[0]["relative_pose_max_abs"] <= 0.05 + 1e-6
    assert trace[0]["relative_pose_raw_scale_max_abs"] <= 1.0 + 1e-6


@pytest.mark.parametrize(
    ("policy", "uncertainty_active"),
    [
        ("identity", [False, False]),
        ("pose_mean", [False, False]),
        ("amplitude_only", [False, True]),
        ("joint", [False, True]),
    ],
)
def test_explicit_policy_drives_next_layer_trace_and_proposal_digest(
    policy: str, uncertainty_active: list[bool]
) -> None:
    model = LVSMDecoderOnlyModel(
        _config("joint", policy=policy, num_layers=2)
    )
    model(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    trace = model.last_layerwise_trace
    assert [row["uncertainty_active"] for row in trace] == uncertainty_active
    assert trace[0]["cache_generation"] == 1
    assert trace[1]["cache_generation"] == 2
    assert len(trace[0]["proposal_digest"]) == 64
    assert len(trace[0]["applied_state_digest"]) == 64
    assert "proposal_digest" not in trace[1]


def test_edge_off_trace_keeps_first_proposal_but_disables_writeback() -> None:
    model = LVSMDecoderOnlyModel(
        _config(
            "joint",
            policy="joint",
            num_layers=2,
            edge_mask=(False, True, True, True, True),
        )
    )
    model(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    trace = model.last_layerwise_trace
    assert trace[0]["edge_enabled"] is False
    assert len(trace[0]["proposal_digest"]) == 64
    assert trace[1]["geometry_digest"] == trace[0]["geometry_digest"]


def test_writeback_pair_is_initialization_matched_and_diverges_only_after_update() -> None:
    disabled_config = _config(
        "joint",
        policy="joint",
        num_layers=2,
        edge_mask=(False,) * 5,
        update_parameterization="recurrent_delta",
    )
    enabled_config = _config(
        "joint",
        policy="joint",
        num_layers=2,
        edge_mask=(True,) * 5,
        update_parameterization="recurrent_delta",
    )

    torch.manual_seed(29)
    disabled = LVSMDecoderOnlyModel(disabled_config)
    torch.manual_seed(29)
    enabled = LVSMDecoderOnlyModel(enabled_config)

    disabled_parameters = dict(disabled.named_parameters())
    enabled_parameters = dict(enabled.named_parameters())
    assert disabled_parameters.keys() == enabled_parameters.keys()
    for name in disabled_parameters:
        assert torch.equal(disabled_parameters[name], enabled_parameters[name]), name

    disabled_state = disabled.state_dict()
    enabled_state = enabled.state_dict()
    assert disabled_state.keys() == enabled_state.keys()
    for name in disabled_state:
        assert torch.equal(disabled_state[name], enabled_state[name]), name

    images = torch.linspace(-1.0, 1.0, 16 * 16 * 3).reshape(1, 1, 16, 16, 3)
    depths = torch.ones(1, 1, 16, 16, 1)
    torch.manual_seed(31)
    disabled_output = disabled(
        images, _cameras(), _cameras(), context_depths=depths
    )
    torch.manual_seed(31)
    enabled_output = enabled(images, _cameras(), _cameras(), context_depths=depths)
    disabled_loss = disabled_output.square().mean()
    enabled_loss = enabled_output.square().mean()

    assert torch.equal(disabled_output, enabled_output)
    assert torch.equal(disabled_loss, enabled_loss)
    disabled_trace = disabled.last_layerwise_trace
    enabled_trace = enabled.last_layerwise_trace
    assert len(disabled_trace[0]["proposal_digest"]) == 64
    assert disabled_trace[0]["proposal_digest"] == enabled_trace[0]["proposal_digest"]
    assert disabled_trace[0]["edge_enabled"] is False
    assert enabled_trace[0]["edge_enabled"] is True

    disabled_loss.backward()
    enabled_loss.backward()
    disabled_head = disabled.layerwise_belief_refiner.side_heads[0]
    enabled_head = enabled.layerwise_belief_refiner.side_heads[0]
    assert disabled_head.pose_projection.weight.grad is None
    assert disabled_head.depth_projection.weight.grad is None
    assert enabled_head.pose_projection.weight.grad is not None
    assert enabled_head.depth_projection.weight.grad is not None
    assert torch.count_nonzero(enabled_head.pose_projection.weight.grad) > 0
    assert torch.count_nonzero(enabled_head.depth_projection.weight.grad) > 0

    for name, disabled_parameter in disabled_parameters.items():
        if name.startswith("layerwise_belief_refiner."):
            continue
        enabled_parameter = enabled_parameters[name]
        if disabled_parameter.grad is None or enabled_parameter.grad is None:
            assert disabled_parameter.grad is None and enabled_parameter.grad is None
        else:
            assert torch.equal(disabled_parameter.grad, enabled_parameter.grad), name

    disabled_optimizer = torch.optim.Adam(disabled.parameters(), lr=1e-3)
    enabled_optimizer = torch.optim.Adam(enabled.parameters(), lr=1e-3)
    disabled_optimizer.step()
    enabled_optimizer.step()

    for name, disabled_parameter in disabled.named_parameters():
        if name.startswith("layerwise_belief_refiner."):
            continue
        assert torch.equal(disabled_parameter, dict(enabled.named_parameters())[name]), name
    assert torch.count_nonzero(disabled_head.pose_projection.weight) == 0
    assert torch.count_nonzero(disabled_head.depth_projection.weight) == 0
    assert not torch.equal(
        disabled_head.pose_projection.weight,
        enabled_head.pose_projection.weight,
    )
    assert not torch.equal(
        disabled_head.depth_projection.weight,
        enabled_head.depth_projection.weight,
    )


@pytest.mark.parametrize("family", ("ray", "ray_point", "segment_bounds"))
def test_every_fixed_rope_family_runs_the_complete_layerwise_forward(
    family: str,
) -> None:
    torch.manual_seed(7)
    model = LVSMDecoderOnlyModel(
        _config("joint", num_layers=2, family=family)
    )
    output = model(
        torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras()
    )
    assert output.shape == (1, 1, 16, 16, 3)
    assert [row["uncertainty_active"] for row in model.last_layerwise_trace] == [
        False,
        True,
    ]
    if family == "ray":
        assert model.attention.ray_head_ids == list(range(4))
        assert model.attention.point_head_ids == []
    elif family == "ray_point":
        assert len(model.attention.ray_head_ids) == 2
        assert len(model.attention.point_head_ids) == 2
    else:
        assert len(model.attention.segment_head_ids) == 4


def test_layerwise_joint_backward_reaches_mean_and_scale_branches() -> None:
    torch.manual_seed(4)
    model = LVSMDecoderOnlyModel(_config("joint", num_layers=2))
    imgs = torch.randn(1, 1, 16, 16, 3)
    depths = torch.ones(1, 1, 16, 16, 1)
    output = model(imgs, _cameras(), _cameras(), context_depths=depths)
    output.square().mean().backward()
    head = model.layerwise_belief_refiner.side_heads[0]
    assert head.pose_projection.weight.grad is not None
    assert head.depth_projection.weight.grad is not None
    assert torch.isfinite(head.pose_projection.weight.grad).all()
    assert torch.isfinite(head.depth_projection.weight.grad).all()
    assert torch.count_nonzero(head.pose_projection.weight.grad) > 0
    assert torch.count_nonzero(head.depth_projection.weight.grad) > 0


def test_layerwise_mean_only_keeps_same_timing_but_disables_uncertainty() -> None:
    torch.manual_seed(5)
    model = LVSMDecoderOnlyModel(_config("mean_only", num_layers=2))
    model(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    assert [row["uncertainty_active"] for row in model.last_layerwise_trace] == [
        False,
        False,
    ]


def test_layerwise_joint_exposes_eval_only_amplitude_intervention() -> None:
    model = LVSMDecoderOnlyModel(
        _config("joint", num_layers=2, amplitude_intervention="zero")
    )
    model(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    assert [row["amplitude_intervention"] for row in model.last_layerwise_trace] == [
        "off",
        "zero",
    ]


def test_bare_joint_preserves_missing_depth_and_legacy_mode_is_explicit() -> None:
    canonical = LVSMDecoderOnlyModel(_config("joint", num_layers=2))
    canonical(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    assert canonical.last_layerwise_trace[1]["available_depth_tokens"] == 0

    legacy = LVSMDecoderOnlyModel(
        _config("joint", num_layers=2, legacy_compatibility=True)
    )
    legacy(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    assert legacy.last_layerwise_trace[1]["available_depth_tokens"] == 8


def test_layerwise_trace_is_opt_in_and_config_resolution_is_non_mutating() -> None:
    config = _config("joint", num_layers=2, trace_enabled=False)
    assert config.uncertainty_strategy is None
    assert config.geometry_mode == "dual"
    model = LVSMDecoderOnlyModel(config)
    model(torch.randn(1, 1, 16, 16, 3), _cameras(), _cameras())
    assert model.last_layerwise_trace == []
    assert config.uncertainty_strategy is None
    assert config.geometry_mode == "dual"
    assert model.config.uncertainty_strategy == "linearized_shared_sample"
    assert model.config.geometry_mode == "dual"


def test_layerwise_rejects_conflicting_topology_and_legacy_policy() -> None:
    conflict = _config("joint", num_layers=2, family="ray")
    conflict.geometry_mode = "segment"
    with pytest.raises(ValueError, match="requires geometry_mode=ray_only"):
        LVSMDecoderOnlyModel(conflict)

    legacy_with_policy = _config(
        "joint", num_layers=2, policy="joint", legacy_compatibility=True
    )
    with pytest.raises(ValueError, match="legacy compatibility"):
        LVSMDecoderOnlyModel(legacy_with_policy)

    checkpointed = _config("joint", num_layers=2)
    checkpointed.encoder.checkpointing = True
    with pytest.raises(ValueError, match="checkpointing=False"):
        LVSMDecoderOnlyModel(checkpointed)
