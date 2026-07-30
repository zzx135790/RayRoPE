"""Cross-repository LVSM integration tests for FlagRoPE segment modes."""

import torch
import torch.nn.functional as F

from nvs.lvsm import (
    LVSMDecoderOnlyModel,
    LVSMDecoderOnlyModelConfig,
    _physical_uncertainty_from_pose_sigma,
)
from pos_enc.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)


def _encoder():
    return TransformerEncoderConfig(
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
            predict_d="predict_dsig",
            init_depth=0.0,
            init_sigma=6.0,
        ),
        num_layers=1,
        input_norm=True,
        output_norm=True,
        checkpointing=False,
    )


def _config(geometry_mode, depth_phase_mode="uniform_sinc"):
    return LVSMDecoderOnlyModelConfig(
        ref_views=4,
        tar_views=1,
        encoder=_encoder(),
        img_shape=(32, 32, 3),
        cam_shape=(32, 32, 6),
        patch_size=8,
        pos_enc="flag_rope",
        depth_type="predict_dsig",
        init_d=0.0,
        init_sig=6.0,
        geometry_mode=geometry_mode,
        depth_phase_mode=depth_phase_mode,
        segment_pair_allocation=(6, 6, 6, 6),
        scene_scale_value=1.0,
        freq_min_lambda=0.1,
        freq_max_lambda=2.0,
    )


def test_lvsm_threads_segment_modes_into_flag_rope_attention():
    model = LVSMDecoderOnlyModel(_config("segment", "mean"))

    flag_config = model.attention.config
    assert flag_config.geometry_mode == "segment"
    assert flag_config.depth_phase_mode == "mean"
    assert flag_config.num_segment_heads == 4
    assert flag_config.num_ray_heads == 0
    assert flag_config.num_point_heads == 0
    assert flag_config.segment_pair_allocation == (6, 6, 6, 6)


def test_lvsm_threads_nondefault_segment_pair_allocation():
    config = _config("segment")
    config.segment_pair_allocation = (3, 7, 5, 9)

    model = LVSMDecoderOnlyModel(config)

    assert model.attention.config.segment_pair_allocation == (3, 7, 5, 9)


def test_lvsm_threads_ray_only_mode_without_legacy_head_overrides():
    model = LVSMDecoderOnlyModel(_config("ray_only"))

    flag_config = model.attention.config
    assert flag_config.geometry_mode == "ray_only"
    assert flag_config.num_ray_heads == 4
    assert flag_config.num_point_heads == 0


def test_five_arm_modes_keep_parameter_shapes_and_seeded_initialization_equal():
    arms = [
        ("dual", "uniform_sinc"),
        ("dual", "mean"),
        ("ray_only", "uniform_sinc"),
        ("segment", "uniform_sinc"),
        ("segment", "mean"),
    ]
    states = []
    for geometry_mode, depth_phase_mode in arms:
        torch.manual_seed(123)
        model = LVSMDecoderOnlyModel(_config(geometry_mode, depth_phase_mode))
        states.append({name: value.detach().clone() for name, value in model.named_parameters()})

    reference = states[0]
    for state in states[1:]:
        assert state.keys() == reference.keys()
        for name in reference:
            assert state[name].shape == reference[name].shape
            assert torch.equal(state[name], reference[name]), name


def test_lvsm_threads_owner_shared_configuration_into_flag_rope():
    config = _config("dual")
    config.head_aware_frequency_layout = True
    config.rope_family = "ray_point"
    config.uncertainty_strategy = "linearized_shared_sample"
    config.nonlinear_sample_chunk_size = 7

    model = LVSMDecoderOnlyModel(config)

    assert model.attention.config.rope_family == "ray_point"
    assert (
        model.attention.config.uncertainty_strategy
        == "linearized_shared_sample"
    )
    assert model.attention.config.nonlinear_sample_chunk_size == 7


def test_lvsm_converts_token_expanded_pose_sigma_back_to_camera_owners():
    rot_camera = torch.tensor([[0.1, 0.2, 0.0]])
    trans_camera = torch.tensor([[1.0, 2.0, 0.0]])
    patches = 4
    sigma = {
        "pose_rot": rot_camera.repeat_interleave(patches, dim=1).unsqueeze(1),
        "pose_trans": trans_camera.repeat_interleave(patches, dim=1).unsqueeze(1),
    }

    uncertainty = _physical_uncertainty_from_pose_sigma(
        sigma,
        batch_size=2,
        camera_count=3,
        num_patches=patches,
    )

    assert uncertainty.pose is not None
    assert uncertainty.pose.scale.shape == (2, 3, 6)
    assert torch.equal(uncertainty.pose.scale[0, :, 0], rot_camera[0])
    assert torch.equal(uncertainty.pose.scale[0, :, 3], trans_camera[0])
    assert torch.equal(uncertainty.pose.scale[0], uncertainty.pose.scale[1])
