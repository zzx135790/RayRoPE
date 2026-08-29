"""
Implementation of https://arxiv.org/abs/2410.17242
"""

from dataclasses import dataclass, field
import copy
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from pos_enc.xy_rope import xyRopeDotProductAttention
from pos_enc.rayrope import RayRoPE_DotProductAttention
from pos_enc.rayrope_nosig import RayRoPE_NoSig_DotProductAttention
from pos_enc.rope_global_ray import RoPE_GlobalRay_DotProductAttention
from pos_enc.timing_utils import time_block

from pos_enc.prope import PropeDotProductAttention
from pos_enc.utils.functional import (
    Camera,
    camera_to_raymap,
    patchify,
    raymap_to_plucker,
    unpatchify,
)
from pos_enc.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)
from nvs.depth_width_observability import DepthWidthObservabilityRecorder
from nvs.depth_width_calibration import DepthWidthCalibrationRecorder
from nvs.depth_width_scale_calibration import DepthWidthScaleCalibrationRecorder


def _physical_uncertainty_from_pose_sigma(
    sigma_overrides: dict,
    *,
    batch_size: int,
    camera_count: int,
    num_patches: int,
):
    """Convert legacy token-expanded pose sigma to camera-owned uncertainty."""

    from tokenmap.models.scene.probabilistic_flag_rope.uncertainty_source import (
        GaussianUncertainty,
        PhysicalUncertainty,
    )

    expected_tokens = camera_count * num_patches
    rot_token = sigma_overrides["pose_rot"]
    trans_token = sigma_overrides["pose_trans"]
    if rot_token.shape[-1] != expected_tokens:
        raise ValueError("pose sigma token count does not match FlagRoPE geometry")
    if rot_token.shape[0] != batch_size:
        if batch_size % rot_token.shape[0] != 0:
            raise ValueError("pose sigma batch cannot be expanded to LVSM target batch")
        repeats = batch_size // rot_token.shape[0]
        rot_token = rot_token.repeat_interleave(repeats, dim=0)
        trans_token = trans_token.repeat_interleave(repeats, dim=0)
    rot_camera = rot_token[:, 0, ::num_patches]
    trans_camera = trans_token[:, 0, ::num_patches]
    pose_scale = torch.cat(
        (
            rot_camera[..., None].expand(-1, -1, 3),
            trans_camera[..., None].expand(-1, -1, 3),
        ),
        dim=-1,
    )
    return PhysicalUncertainty(
        pose=GaussianUncertainty(scale=pose_scale, owner="camera")
    )


def _transform_depth_uncertainty(
    predicted_d: torch.Tensor, transform: str, num_patches: int
) -> torch.Tensor:
    """Apply an eval-only width intervention while preserving log-depth centres."""

    if transform == "true":
        return predicted_d
    if predicted_d.ndim != 3 or predicted_d.shape[-1] != 2:
        raise ValueError("predicted depth must have shape [B,N,2]")
    if transform not in ("zero", "permute_within_camera"):
        raise ValueError("unsupported depth uncertainty transform")
    result = predicted_d.clone()
    if transform == "zero":
        result[..., 1] = 0
        return result
    if result.shape[1] % num_patches:
        raise ValueError("token count must be divisible by num_patches")
    camera_count = result.shape[1] // num_patches
    widths = result[..., 1].reshape(result.shape[0], camera_count, num_patches)
    result[..., 1] = torch.roll(widths, shifts=1, dims=2).reshape(result.shape[0], -1)
    return result


def _sample_layerwise_depth_tokens(
    context_depths: Optional[torch.Tensor],
    *,
    target_views: int,
    patch_size: int,
    reference_views: int,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Mask-aware patch means for external reference depth.

    Validity is computed before any numerical replacement.  A patch is
    available when at least one finite, positive source pixel is present; target
    tokens and empty reference patches remain explicit zero-length geometry.
    """

    if context_depths is None:
        return None, None
    if context_depths.ndim != 5 or context_depths.shape[-1] != 1:
        raise ValueError("context_depths must have shape [B,V,H,W,1]")
    batch, views, height, width, _ = context_depths.shape
    if views != reference_views:
        raise ValueError("context depth view count does not match reference cameras")
    if height % patch_size or width % patch_size:
        raise ValueError("context depth resolution must be divisible by patch_size")

    values = context_depths[..., 0]
    valid = torch.isfinite(values) & (values > 0)
    safe_values = torch.where(valid, values, torch.zeros_like(values))
    flat_values = safe_values.reshape(batch * views, 1, height, width)
    flat_valid = valid.to(values.dtype).reshape(batch * views, 1, height, width)
    value_mean = F.avg_pool2d(flat_values, patch_size, stride=patch_size)
    valid_fraction = F.avg_pool2d(flat_valid, patch_size, stride=patch_size)
    patch_available = valid_fraction > 0
    patch_depth = value_mean / valid_fraction.clamp_min(
        torch.finfo(values.dtype).eps
    )
    patch_depth = torch.where(
        patch_available, patch_depth, torch.ones_like(patch_depth)
    )
    patch_depth = patch_depth.reshape(batch, views, -1)
    patch_available = patch_available.reshape(batch, views, -1)
    patch_count = patch_depth.shape[-1]

    repeated_depth = repeat(
        patch_depth, "b v p -> (b vt) v p", vt=target_views
    )
    repeated_available = repeat(
        patch_available, "b v p -> (b vt) v p", vt=target_views
    )
    target_depth = torch.ones(
        batch * target_views,
        1,
        patch_count,
        device=values.device,
        dtype=values.dtype,
    )
    target_available = torch.zeros(
        batch * target_views,
        1,
        patch_count,
        device=values.device,
        dtype=torch.bool,
    )
    return (
        torch.cat((repeated_depth, target_depth), dim=1).reshape(
            batch * target_views, -1
        ),
        torch.cat((repeated_available, target_available), dim=1).reshape(
            batch * target_views, -1
        ),
    )


def _prepare_depth_input(
    context_depths: Optional[torch.Tensor],
    *,
    reference_images: torch.Tensor,
) -> torch.Tensor:
    """Build a finite inverse-depth input while preserving missing-depth holes."""

    if reference_images.ndim != 5 or reference_images.shape[-1] < 1:
        raise ValueError("reference_images must have shape [B,V,H,W,C]")
    batch, views, height, width = reference_images.shape[:4]
    if context_depths is None:
        return torch.zeros(
            batch,
            views,
            height,
            width,
            1,
            device=reference_images.device,
            dtype=reference_images.dtype,
        )
    if context_depths.ndim != 5 or context_depths.shape[-1] != 1:
        raise ValueError("context_depths must have shape [B,V,H,W,1]")
    if tuple(context_depths.shape[:4]) != (batch, views, height, width):
        raise ValueError("context_depths must match reference image dimensions")
    depths = context_depths.to(
        device=reference_images.device, dtype=reference_images.dtype
    )
    valid = torch.isfinite(depths) & (depths > 0)
    safe_depths = torch.where(valid, depths, torch.ones_like(depths))
    inverse = safe_depths.reciprocal()
    # Zero is the finite input-channel sentinel for an unavailable depth.  The
    # geometry path carries the authoritative availability mask separately.
    return torch.where(valid, inverse, torch.zeros_like(inverse))


@dataclass
class LVSMDecoderOnlyModelConfig:

    ref_views: int
    tar_views: int = 1

    encoder: TransformerEncoderConfig = field(
        default_factory=lambda: TransformerEncoderConfig(
            layer=TransformerEncoderLayerConfig(
                d_model=768,
                nhead=16,
                dim_feedforward=3072,
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
                predict_d='none',
            ),
            num_layers=6,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
    )

    img_shape: Tuple[int, ...] = (256, 256, 3)
    cam_shape: Tuple[int, ...] = (256, 256, 6)
    patch_size: int = 8

    # How the input rays are encoded.
    ray_encoding: Literal["plucker", "camray", "none", "raymap"] = "plucker"

    pos_enc: str = "d_pj+0_3d"
    num_rays_per_patch: int = 3
    freq_base: float = 3.0
    disable_vo: bool = False
    depth_type: str = "none"
    init_d: float = 0.0
    init_sig: float = 3.0
    geometry_mode: Literal["dual", "ray_only", "segment"] = "dual"
    depth_phase_mode: Literal["mean", "uniform_sinc"] = "uniform_sinc"
    segment_pair_allocation: Tuple[int, int, int, int] = (6, 6, 6, 6)
    head_aware_frequency_layout: bool = False
    frequency_bank_sharing: Literal["per_head", "pair", "shared"] = "per_head"
    frequency_allocation: Literal["interleaved", "contiguous"] = "interleaved"
    frequency_axis_layout: Literal["tensor_product", "round_robin"] = "tensor_product"
    segment_head_allocation: Tuple[int, int, int, int] = (4, 4, 4, 4)
    segment_endpoint_bounds: bool = False
    rope_family: Optional[Literal["ray", "ray_point", "segment_bounds"]] = None
    uncertainty_strategy: Optional[
        Literal["none", "linearized_shared_sample", "nonlinear_shared_sample"]
    ] = None
    # Execution-only bound for vectorized nonlinear owner samples.
    nonlinear_sample_chunk_size: int = 16
    # Source calibration for predict_dsig's raw log-depth half-width. The
    # FlagRoPE source boundary applies this once for every runtime depth path.
    depth_width_calibrator: Literal["identity", "raw_multiplier"] = "identity"
    depth_width_scale: float = 1.0
    
    denc_type: str = "d"  # "d" or "inv_d" or "asinh_d"
    depth_input: bool = False # concat context depth map to ref input

    # flag_rope scene-scale normalisation ablation knobs (no effect for RayRoPE).
    # scene_scale_source: "context"=median of real ref depths (fixed), "ones"=1.0 (bug repro).
    # normalize_transform: True=fixed (t_q/s, no double /s on moment), False=bug repro.
    scene_scale_source: str = "context"
    normalize_transform: bool = True

    # flag_rope head allocation override (no effect for RayRoPE). When None, the
    # default split is content=0, ray=ceil(nhead/2), point=floor(nhead/2). Setting
    # num_point_heads=0 yields a pure ray-head ablation (point head removed).
    num_ray_heads: Optional[int] = None
    num_point_heads: Optional[int] = None
    # flag_rope scene-scale calibration override (no effect for RayRoPE). When not
    # None, force a constant s=value per batch (overrides scene_scale_source),
    # so flag_rope can be calibrated on datasets without GT depth (official re10k).
    scene_scale_value: Optional[float] = None
    # flag_rope frequency-bank range override (no effect for RayRoPE). Log-spaced
    # λ∈[min,max], ρ=2π/λ. Default [0.1,12] is designed for O(1) coords with a
    # WIDE dynamic range (DTU ~300). On re10k O(1) with NARROW range (~1) the
    # coarsest block (λ_max) under-aliases dead; narrowing/shifting the band lets
    # us re-fit the bank to the dataset's coord spread.
    freq_min_lambda: float = 0.1
    freq_max_lambda: float = 12.0
    # flag_rope pose-uncertainty awareness (mode C). When True, flag_rope's plain
    # precompute path perturbs the query-frame geometry per frequency block by the
    # pose σ passed to forward() as ``pose_sigma_overrides`` (pose_rot/pose_trans
    # only; depth stays with predict_dsig). No effect for RayRoPE.
    use_pose_uncertainty: bool = False
    # Soft cap on mode C pose-perturbation norm (None=unbounded). Prevents rare
    # degenerate-batch NaN. No effect for RayRoPE / when use_pose_uncertainty=False.
    pose_delta_cap: Optional[float] = None
    # Model A (mode C only): couple the query-frame pose perturbation to the
    # source-camera perturbation (same δ_c for both roles of camera c, Q=C) so
    # same-camera pairs cancel exactly. False (default) = Model B, which leaks
    # pose noise into same-camera attention. No effect unless use_pose_uncertainty.
    pose_query_coupled: bool = False
    # flag_rope X5: analytic CF decay (non-pairwise, Flash-fusable). Requires
    # pose_query_coupled=True (Model A). Mutually exclusive with use_pose_uncertainty.
    use_pose_uncertainty_cf: bool = False
    # flag_rope bilateral CF (NBV axis ② arm): also marginalise the query-side
    # pose in the CF path (J_source+J_query). Same-cam decay=1 by invariance
    # (no hard mask); cross-cam Var doubles vs source-only. Defaults False to
    # preserve X5/flag_aware_CF source-only behaviour. See FlagRoPEConfig.
    use_bilateral_cf: bool = False
    # ── mode D: learnable recurrent (μ,σ) uncertainty (flag_rope only) ──
    # The model SELF-ESTIMATES pose(+depth) (μ,σ) and refines them recurrently
    # across 24 layers via a tied per-layer Δ head (stationary Markov kernel).
    # μ₀=0 (input pose is the mean); σ₀=init (constant or true injected σ).
    # Depth is unified into the 7-state (predict_d='none'; the point sinc
    # integral consumes predicted_d=[μ_d,σ_d] from the recurrent state).
    # Mutually exclusive with use_pose_uncertainty (mode C tells σ, mode D
    # predicts σ). See flag_config.FlagRoPEConfig for field docs.
    use_recurrent_uncertainty: bool = False
    pose_mu_learnable: bool = True
    pose_sigma_init: str = "constant"        # "constant" | "true"
    pose_sigma_init_value: float = 1e-2
    recurrent_mu_clamp: Optional[float] = None
    recurrent_sigma_cap: Optional[float] = None
    # Owner-aware post-block recurrence. Layer 0 consumes mean geometry only;
    # five untied side heads update the six-layer model for the following layer.
    layerwise_belief_mode: Literal[
        "none", "mean_only", "joint", "identity", "pose_mean", "amplitude_only"
    ] = "none"
    layerwise_belief_policy: Optional[
        Literal["identity", "pose_mean", "amplitude_only", "joint"]
    ] = None
    layerwise_update_parameterization: Literal[
        "bounded_residual", "recurrent_delta"
    ] = "bounded_residual"
    layerwise_edge_mask: tuple[bool, bool, bool, bool, bool] = (True, True, True, True, True)
    layerwise_freeze_pose: bool = False
    layerwise_freeze_amplitude: bool = False
    layerwise_pose_intervention: Literal["true", "permute_within_owner"] = "true"
    layerwise_pose_scale_init: float = 1e-2
    layerwise_depth_width_scale: float = 1.0 / 512.0
    layerwise_rotation_cap: float = 0.05
    layerwise_translation_cap: float = 0.05
    layerwise_log_depth_cap: float = 0.5
    layerwise_raw_scale_cap: float = 1.0
    # Evaluation-only counterfactual applied after the side head has produced
    # positive pose/depth amplitudes. It never changes the mean-state update.
    layerwise_amplitude_intervention: Literal[
        "true", "zero", "scale_half", "scale_double", "permute_within_owner"
    ] = "true"
    # New layerwise controls remain in the layerwise block; the pre-existing
    # ``prope_impl`` field stays last for positional-config compatibility.
    layerwise_legacy_compatibility: bool = False
    # Detailed per-layer CPU digests are diagnostic-only and disabled for
    # normal training to avoid repeated GPU-to-CPU synchronization.
    layerwise_trace_enabled: bool = False
    # Timing configuration
    timing_enabled: bool = False
    # Select the vendored default or the locked official PRoPE harness.
    # Kept last to preserve all pre-existing positional config arguments.
    prope_impl: Literal["vendored", "official"] = "vendored"


class LVSMDecoderOnlyModel(nn.Module):
    def __init__(self, config: LVSMDecoderOnlyModelConfig):
        super().__init__()
        # Resolve layerwise defaults on a private copy.  Constructing a model
        # must not mutate a caller-owned config (which may also configure a
        # baseline model or be reused for a second arm).
        self.requested_config = copy.deepcopy(config)
        config = copy.deepcopy(config)
        self.config = config
        self.layerwise_belief_refiner = None
        self.last_layerwise_trace: list[dict[str, int | bool | str | float]] = []

        layerwise_active = (
            config.layerwise_belief_mode != "none"
            or config.layerwise_belief_policy is not None
        )
        if (
            not layerwise_active
            and config.layerwise_amplitude_intervention != "true"
        ):
            raise ValueError(
                "layer-wise amplitude intervention requires layer-wise refinement"
            )
        if layerwise_active:
            if config.pos_enc != "flag_rope":
                raise ValueError("layer-wise belief refinement requires flag_rope")
            if config.rope_family is None:
                raise ValueError("layer-wise belief refinement requires rope_family")
            if config.use_recurrent_uncertainty:
                raise ValueError("layer-wise refinement cannot be mixed with Mode D")
            if config.encoder.checkpointing:
                raise ValueError("layer-wise refinement currently requires checkpointing=False")
            if config.uncertainty_strategy not in (None, "linearized_shared_sample"):
                raise ValueError(
                    "layer-wise refinement trains with linearized_shared_sample"
                )
            if config.layerwise_legacy_compatibility and (
                config.layerwise_belief_policy is not None
            ):
                raise ValueError(
                    "layerwise legacy compatibility requires an implicit policy"
                )
            expected_geometry = {
                "ray": "ray_only",
                "ray_point": "dual",
                "segment_bounds": "segment",
            }[config.rope_family]
            # ``dual`` is the historical default for all families.  It is
            # accepted as an implicit request, while an explicit incompatible
            # family topology is rejected instead of silently rewritten.
            if config.geometry_mode not in ("dual", expected_geometry):
                raise ValueError(
                    f"layerwise {config.rope_family} requires geometry_mode={expected_geometry}"
                )
            config.uncertainty_strategy = "linearized_shared_sample"
            config.head_aware_frequency_layout = True
            config.depth_width_calibrator = "raw_multiplier"
            config.depth_width_scale = config.layerwise_depth_width_scale
            config.geometry_mode = expected_geometry
            if config.rope_family == "ray_point":
                expected_ray_heads = config.encoder.layer.nhead // 2
                expected_point_heads = (
                    config.encoder.layer.nhead - expected_ray_heads
                )
                if config.num_ray_heads is not None and (
                    config.num_ray_heads != expected_ray_heads
                ):
                    raise ValueError(
                        "layerwise ray_point fixes num_ray_heads to half the attention heads"
                    )
                if config.num_point_heads is not None and (
                    config.num_point_heads != expected_point_heads
                ):
                    raise ValueError(
                        "layerwise ray_point fixes num_point_heads to the remaining heads"
                    )
                config.num_ray_heads = config.encoder.layer.nhead // 2
                config.num_point_heads = (
                    config.encoder.layer.nhead - config.num_ray_heads
                )
            elif config.rope_family == "segment_bounds":
                config.segment_endpoint_bounds = True
                if config.encoder.layer.nhead == 16:
                    config.segment_head_allocation = (6, 6, 2, 2)
                elif config.encoder.layer.nhead >= 4:
                    remaining = config.encoder.layer.nhead - 2
                    config.segment_head_allocation = (
                        (remaining + 1) // 2,
                        remaining // 2,
                        1,
                        1,
                    )
                else:
                    raise ValueError(
                        "segment_bounds requires at least four attention heads"
                    )
        
        head_dim = config.encoder.layer.d_model // config.encoder.layer.nhead

        if self.config.pos_enc in ["global-0+inf", "global-0+d"]:
            self.attention = RoPE_GlobalRay_DotProductAttention(
                head_dim=head_dim,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                pos_enc_type=self.config.pos_enc,
                num_rays_per_patch=self.config.num_rays_per_patch,
                freq_base=self.config.freq_base,
            )
        elif ('d_pj' in self.config.pos_enc or \
            'd_3d' in self.config.pos_enc) and \
            'predict_dsig' in self.config.depth_type:
            self.attention = RayRoPE_DotProductAttention(
                head_dim=head_dim,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                pos_enc_type=self.config.pos_enc,
                num_rays_per_patch=self.config.num_rays_per_patch,
                depth_type=self.config.depth_type,
                denc_type=self.config.denc_type,
                freq_base=self.config.freq_base,
                apply_vo=(not self.config.disable_vo),
            )
        elif ('d_pj' in self.config.pos_enc or \
            'd_3d' in self.config.pos_enc) and \
            'predict_d' in self.config.depth_type:
            self.attention = RayRoPE_NoSig_DotProductAttention(
                head_dim=head_dim,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
                pos_enc_type=self.config.pos_enc,
                num_rays_per_patch=self.config.num_rays_per_patch,
                depth_type=self.config.depth_type,
                denc_type=self.config.denc_type,
                freq_base=self.config.freq_base,
                apply_vo=(not self.config.disable_vo),
            )
        elif self.config.pos_enc == "xy_rope":
            self.attention = xyRopeDotProductAttention(
                head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
            )
        elif self.config.pos_enc == "none":
            self.attention = None
        elif self.config.pos_enc == "flag_rope":
            # FlagRoPE (Plücker ray + 3D point dual RoPE) as a RayRoPE-compatible
            # sdpa_fn. Lazy import so RayRoPE stays usable without `tokenmap`.
            # All heads are geometric (content=0) to align with RayRoPE
            # d_pj+0_3d. Depth uncertainty comes from the per-layer predicted_d
            # (predict_dsig); pose/K uncertainty is off (use_uncertainty_perturbation=False).
            from tokenmap.models.scene.probabilistic_flag_rope.sdpa_adapter_multi_query import (
                FlagRoPEMultiQuerySdpaAttention,
            )
            from tokenmap.models.scene.probabilistic_flag_rope.flag_config import (
                FlagRoPEConfig,
            )
            nhead = config.encoder.layer.nhead
            d_model = config.encoder.layer.d_model
            n_geo = nhead  # content=0 → all heads are geometric
            head_kwargs = {}
            if config.geometry_mode == "dual":
                # Preserve the existing ray/point override surface for dual
                # baselines and historical pure-ray ablations.
                head_kwargs = {
                    "num_content_heads": 0,
                    "num_ray_heads": (
                        config.num_ray_heads
                        if config.num_ray_heads is not None
                        else (n_geo + 1) // 2
                    ),
                    "num_point_heads": (
                        config.num_point_heads
                        if config.num_point_heads is not None
                        else n_geo // 2
                    ),
                }
            flag_cfg = FlagRoPEConfig.from_d_model_and_num_heads(
                d_model=d_model,
                num_heads=nhead,
                geometry_mode=self.config.geometry_mode,
                depth_phase_mode=self.config.depth_phase_mode,
                segment_pair_allocation=self.config.segment_pair_allocation,
                head_aware_frequency_layout=self.config.head_aware_frequency_layout,
                frequency_bank_sharing=self.config.frequency_bank_sharing,
                frequency_allocation=self.config.frequency_allocation,
                frequency_axis_layout=self.config.frequency_axis_layout,
                segment_head_allocation=self.config.segment_head_allocation,
                segment_source_target=self.config.segment_endpoint_bounds,
                rope_family=self.config.rope_family,
                uncertainty_strategy=self.config.uncertainty_strategy,
                nonlinear_sample_chunk_size=self.config.nonlinear_sample_chunk_size,
                depth_width_calibrator=self.config.depth_width_calibrator,
                depth_width_scale=self.config.depth_width_scale,
                use_uncertainty_perturbation=False,
                scene_scale_source=self.config.scene_scale_source,
                normalize_transform=self.config.normalize_transform,
                scene_scale_value=self.config.scene_scale_value,
                frequency_min_lambda=self.config.freq_min_lambda,
                frequency_max_lambda=self.config.freq_max_lambda,
                use_pose_uncertainty=self.config.use_pose_uncertainty,
                pose_delta_cap=self.config.pose_delta_cap,
                pose_query_coupled=self.config.pose_query_coupled,
                use_pose_uncertainty_cf=self.config.use_pose_uncertainty_cf,
                use_bilateral_cf=self.config.use_bilateral_cf,
                use_recurrent_uncertainty=self.config.use_recurrent_uncertainty,
                pose_mu_learnable=self.config.pose_mu_learnable,
                pose_sigma_init=self.config.pose_sigma_init,
                pose_sigma_init_value=self.config.pose_sigma_init_value,
                recurrent_mu_clamp=self.config.recurrent_mu_clamp,
                recurrent_sigma_cap=self.config.recurrent_sigma_cap,
                recurrent_init_depth=self.config.init_d,
                recurrent_init_sigma=self.config.init_sig,
                **head_kwargs,
            )
            self.attention = FlagRoPEMultiQuerySdpaAttention(
                config=flag_cfg,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
            )
        else:
            attention_class = PropeDotProductAttention
            if self.config.pos_enc == "prope":
                if self.config.prope_impl not in ("vendored", "official"):
                    raise ValueError(
                        "prope_impl must be 'vendored' or 'official', got "
                        f"{self.config.prope_impl!r}"
                    )
                if self.config.prope_impl == "official":
                    from nvs.official_prope import load_official_prope_attention

                    attention_class = load_official_prope_attention
            self.attention = attention_class(
                head_dim=config.encoder.layer.d_model // config.encoder.layer.nhead,
                # cameras=config.ref_views + config.tar_views,
                patches_x=config.img_shape[1] // config.patch_size,
                patches_y=config.img_shape[0] // config.patch_size,
                image_width=config.img_shape[1],
                image_height=config.img_shape[0],
            )

        assert (
            config.cam_shape[:2] == config.img_shape[:2]
        ), f"{config.cam_shape[:2]} != {config.img_shape[:2]}"

        if config.ray_encoding == "none":
            shared_rays = torch.randn(config.cam_shape)
            self.shared_rays = nn.Parameter(shared_rays, requires_grad=False)

        # query tokenizer encodes tar_cam
        self.query_tokenizer = nn.Linear(
            config.cam_shape[-1] * config.patch_size**2,
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )
        # input tokenizer encodes ref_img and ref_cam
        self.input_tokenizer = nn.Linear(
            (
                config.img_shape[-1] * config.patch_size**2
                + config.cam_shape[-1] * config.patch_size**2
                + int(config.depth_input) * 1 * config.patch_size**2
            ),
            config.encoder.layer.d_model,
            bias=config.encoder.layer.bias,
        )

        if self.config.depth_type in ['predict_d', 'known+predict_d']:
            self.config.encoder.layer.predict_d = 'predict_d'
        elif self.config.depth_type in ['predict_dsig', 'known+predict_dsig']:
            self.config.encoder.layer.predict_d = 'predict_dsig'
        elif self.config.depth_type in ['dsig_perhead', 'known+dsig_perhead']:
            self.config.encoder.layer.predict_d = 'dsig_perhead'
        self.config.encoder.layer.init_depth = self.config.init_d
        self.config.encoder.layer.init_sigma = self.config.init_sig
        # mode D: depth comes from the recurrent 7-state, so disable the
        # separate predict_dsig depth head (avoid two depth heads competing)
        # and enable the per-layer Δ head (tied across clones). predict_d is
        # forced to 'none' regardless of depth_type when mode D is on.
        if self.config.use_recurrent_uncertainty:
            self.config.encoder.layer.predict_d = 'none'
            self.config.encoder.layer.predict_delta = True
        elif layerwise_active:
            self.config.encoder.layer.predict_d = 'none'
            self.config.encoder.layer.predict_delta = False
        
        self.encoder = self.config.encoder.setup()

        if layerwise_active:
            from tokenmap.models.scene.probabilistic_flag_rope.layerwise_belief import (
                LayerwiseBeliefConfig,
                LayerwiseBeliefRefiner,
            )

            self.layerwise_belief_refiner = LayerwiseBeliefRefiner(
                    LayerwiseBeliefConfig(
                    d_model=config.encoder.layer.d_model,
                    num_layers=config.encoder.num_layers,
                    num_patches=(config.img_shape[0] // config.patch_size)
                    * (config.img_shape[1] // config.patch_size),
                    mode=config.layerwise_belief_mode,
                    policy=config.layerwise_belief_policy,
                    update_parameterization=config.layerwise_update_parameterization,
                    rope_family=config.rope_family,
                    pose_scale_init=config.layerwise_pose_scale_init,
                    depth_raw_width_init=config.init_sig,
                    depth_width_scale=config.layerwise_depth_width_scale,
                    rotation_cap=config.layerwise_rotation_cap,
                    translation_cap=config.layerwise_translation_cap,
                    log_depth_cap=config.layerwise_log_depth_cap,
                    raw_scale_cap=config.layerwise_raw_scale_cap,
                    amplitude_intervention=config.layerwise_amplitude_intervention,
                    pose_intervention=config.layerwise_pose_intervention,
                    legacy_compatibility=config.layerwise_legacy_compatibility,
                )
            )

        self.output_layer = nn.Linear(
            config.encoder.layer.d_model,
            config.img_shape[-1] * config.patch_size**2,
            bias=config.encoder.layer.bias,
        )
        self.init_weights()

    def init_weights(self):
        for idx, layer in enumerate(self.encoder.layers):
            layer.apply(self.init_layer_weights(idx))

    def init_layer_weights(self, idx):
        # LVMS Paper A.1:
        # "We initialize the model weights with a normal distribution of zero-mean
        # and standard deviation of 0.02/(2 * (idx+ 1)) ** 0.5, where idx means
        # transform layer index."
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.02 / (2 * (idx + 1)) ** 0.5)

        return _init_weights

    def create_rays(self, cams: Camera) -> Tensor:
        """Convert cameras to raymaps.

        Returns:
            rays: [B, V, H, W, C]
        """
        config = self.config
        batch_size, v = cams.camtoworld.shape[:2]
        cam_dtype = cams.camtoworld.dtype
        device = cams.camtoworld.device

        if config.ray_encoding == "none":
            rays = repeat(self.shared_rays, "h w c -> b v h w c", b=batch_size, v=v)
        else:
            # Preprocess cameras into rays.
            downscale = config.img_shape[0] // config.cam_shape[0]
            rays = camera_to_raymap(
                Ks=cams.K,
                camtoworlds=(
                    torch.eye(4, dtype=cam_dtype, device=device).broadcast_to(
                        cams.camtoworld.shape
                    )
                    if config.ray_encoding == "camray"
                    else cams.camtoworld
                ),
                height=cams.height,
                width=cams.width,
                downscale=downscale,
            )
            if config.ray_encoding in ["plucker", "camray"]:
                rays = raymap_to_plucker(rays)
            else:
                assert config.ray_encoding == "raymap"
        return rays

    def forward(
        self,
        ref_imgs: Tensor,
        ref_cams: Camera,
        tar_cams: Camera,
        context_depths: Optional[Tensor] = None,
        timing_enabled: bool = False,
        pose_sigma_overrides: Optional[dict] = None,
        pose_sigma_seed: Optional[dict] = None,
        uncertainty_sample_seed: Optional[int] = None,
        depth_uncertainty_transform: str = "true",
        depth_observability_recorder: Optional[
            DepthWidthObservabilityRecorder
            | DepthWidthCalibrationRecorder
            | DepthWidthScaleCalibrationRecorder
        ] = None,
    ) -> Tensor:

        with time_block("preprocess", enabled=timing_enabled):
            # ref_imgs: [B, V1, H, W, C]
            # tar_imgs: [B, V2, H, W, C]
            batch_size, v2 = tar_cams.camtoworld.shape[:2]
            config = self.config

            # Create rays.
            # ref_rays: [B, V1, H, W, C]
            # tar_rays: [B, V2, H, W, C]
            ref_rays = self.create_rays(ref_cams)
            tar_rays = self.create_rays(tar_cams)
            raw_ref_imgs = ref_imgs

            # print(f"before patchify:")
            # print(f"ref_imgs shape: {ref_imgs.shape}, ref_rays shape: {ref_rays.shape}, context_depths_patch shape: {context_depths.shape}")
            # ref_imgs: [B, V1, N1, DIM1]
            ref_imgs = patchify(ref_imgs, config.patch_size)
            # ref_rays: [B, V1, N2, DIM2]
            ref_rays = patchify(ref_rays, config.patch_size)
            # tar_rays: [B, V2, N2, DIM2]
            tar_rays = patchify(tar_rays, config.patch_size)

            # context_depths: [B, V1, N1, 1]
            if self.config.depth_input:
                inverse_depths = _prepare_depth_input(
                    context_depths, reference_images=raw_ref_imgs
                )
                context_depths_patch = patchify(inverse_depths, config.patch_size)

            # Tokenize into
            # x: [B*V2, V1*N1, DIM1]
            # q: [B*V2, N2, DIM2]
            if self.config.depth_input:
                x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays, context_depths_patch], dim=-1))
            else:
                x = self.input_tokenizer(torch.cat([ref_imgs, ref_rays], dim=-1))
            x = repeat(x, "b v1 n d -> (b v2) (v1 n) d", v2=v2)
            q = self.query_tokenizer(tar_rays)
            q = rearrange(q, "b v2 n d -> (b v2) n d")
            q_tokens = q.shape[1]

            # --- Prepare data for geomtry-aware self-attention ---
            ref_c2ws = repeat(ref_cams.camtoworld, "b v1 x y -> (b v2) v1 x y", v2=v2)
            ref_Ks = repeat(ref_cams.K, "b v1 x y -> (b v2) v1 x y", v2=v2)
            tar_c2ws = rearrange(tar_cams.camtoworld, "b v2 x y -> (b v2) 1 x y", v2=v2)
            tar_Ks = rearrange(tar_cams.K, "b v2 x y -> (b v2) 1 x y")
            c2ws = torch.cat([ref_c2ws, tar_c2ws], dim=1)  # [B, N, 4, 4] per camera
            Ks = torch.cat([ref_Ks, tar_Ks], dim=1)  # [B, N, 3, 3] per camera
            viewmats = torch.inverse(c2ws)

        with time_block("precompute_enc", enabled=timing_enabled):
            if  "0_pj" in config.pos_enc or "0_3d" in config.pos_enc \
                or config.pos_enc in ["global-0+inf", "global-0+d"] \
                or config.pos_enc == "flag_rope":

                if context_depths is not None:
                    depths_for_rope = repeat(context_depths, "b v1 h w 1 -> (b v2) v1 h w 1", v2=v2)
                else:
                    depths_for_rope = None
                _pc_kwargs = dict(w2cs=viewmats, Ks=Ks, context_depths=depths_for_rope)
                # flag_rope uncertainty σ routing:
                #   mode C (use_pose_uncertainty): pose_sigma_overrides = the TRUE
                #     injected σ, used to perturb geometry per frequency block.
                #   mode D (use_recurrent_uncertainty): pose_sigma_seed = the TRUE
                #     injected σ, used ONLY to seed σ₀ (teacher-forcing) when
                #     pose_sigma_init=="true"; the per-layer perturbation comes
                #     from the recurrent σ (model self-estimates). At test no σ
                #     is passed (head recovers σ from features).
                # Other pos_enc (RayRoPE) _precompute_and_cache_apply_fns don't
                # accept sigma_overrides.
                if self.config.pos_enc == "flag_rope":
                    shared_strategy = self.config.uncertainty_strategy in (
                        "linearized_shared_sample",
                        "nonlinear_shared_sample",
                    )
                    if shared_strategy and pose_sigma_overrides is not None:
                        _pc_kwargs["uncertainty"] = (
                            _physical_uncertainty_from_pose_sigma(
                                pose_sigma_overrides,
                                batch_size=viewmats.shape[0],
                                camera_count=viewmats.shape[1],
                                num_patches=self.attention.num_patches,
                            )
                        )
                    elif self.config.use_recurrent_uncertainty:
                        if pose_sigma_seed is not None:
                            _pc_kwargs["sigma_overrides"] = pose_sigma_seed
                    elif pose_sigma_overrides is not None:
                        _pc_kwargs["sigma_overrides"] = pose_sigma_overrides
                if self.layerwise_belief_refiner is None:
                    self.attention._precompute_and_cache_apply_fns(**_pc_kwargs)

        attention_call_index = 0

        def sdpa_fn(q, k, v, **sdpa_kwargs):
            nonlocal attention_call_index
            shared_strategy = config.uncertainty_strategy in (
                "linearized_shared_sample",
                "nonlinear_shared_sample",
            )
            if shared_strategy:
                predicted_d = sdpa_kwargs.get("predicted_d")
                if predicted_d is not None:
                    transformed_d = _transform_depth_uncertainty(
                        predicted_d,
                        depth_uncertainty_transform,
                        self.attention.num_patches,
                    )
                    sdpa_kwargs["predicted_d"] = transformed_d
                    if depth_observability_recorder is not None:
                        sdpa_kwargs["depth_observability"] = (
                            depth_observability_recorder.request(
                                layer_index=attention_call_index,
                                original_predicted_d=predicted_d,
                                actual_predicted_d=transformed_d,
                            )
                        )
                elif depth_observability_recorder is not None:
                    raise ValueError(
                        "depth observability requires per-layer predicted depth widths"
                    )
                if uncertainty_sample_seed is not None:
                    generator = torch.Generator(device=q.device)
                    generator.manual_seed(
                        int(uncertainty_sample_seed) + attention_call_index
                    )
                    sdpa_kwargs["generator"] = generator
                attention_call_index += 1
            if config.pos_enc == "gta":
                # GTA is effectively PRoPE without intrinsics.
                return self.attention(q, k, v, viewmats=viewmats, Ks=None, timing_enabled=timing_enabled, **sdpa_kwargs)
            elif config.pos_enc == "prope":
                return self.attention(q, k, v, viewmats=viewmats, Ks=Ks, timing_enabled=timing_enabled, **sdpa_kwargs)
            elif config.pos_enc == "none":
                # Use the default attention.
                out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
                return out
            else:
                return self.attention(q, k, v, timing_enabled=timing_enabled, **sdpa_kwargs)
            
        if config.pos_enc == "none":
            sdpa_fn = F.scaled_dot_product_attention
            
        with time_block("transformer", enabled=timing_enabled):
            # run attentions
            xq = torch.cat([x, q], dim=1)
            layer_controller = None
            if self.layerwise_belief_refiner is not None:
                initial_depth, depth_available = _sample_layerwise_depth_tokens(
                    context_depths,
                    target_views=v2,
                    patch_size=config.patch_size,
                    reference_views=ref_cams.camtoworld.shape[1],
                )
                scene_scale = None
                if config.scene_scale_value is not None:
                    scene_scale = torch.full(
                        (viewmats.shape[0],),
                        float(config.scene_scale_value),
                        device=viewmats.device,
                        dtype=viewmats.dtype,
                    )
                from tokenmap.models.scene.probabilistic_flag_rope import (
                    FlagRoPEAttentionSink,
                    LayerwiseEdgeMask,
                    LayerwiseRuntimeController,
                )

                configured_edges = tuple(config.layerwise_edge_mask)
                expected_edges = config.encoder.num_layers - 1
                if len(configured_edges) != expected_edges:
                    # The historical config default is a five-edge tuple for
                    # the six-layer encoder.  Preserve that input surface for
                    # smaller CPU tests while validating the effective mask.
                    if len(configured_edges) == 5:
                        configured_edges = configured_edges[:expected_edges]
                    else:
                        raise ValueError(
                            "layerwise_edge_mask must have num_layers - 1 entries"
                        )
                edge_mask = LayerwiseEdgeMask(
                    edges=configured_edges,
                    pose_mean=not config.layerwise_freeze_pose,
                    amplitude=not config.layerwise_freeze_amplitude,
                )
                layer_controller = LayerwiseRuntimeController.from_inputs(
                    refiner=self.layerwise_belief_refiner,
                    sink=FlagRoPEAttentionSink(self.attention),
                    w2cs=viewmats,
                    intrinsics=Ks,
                    depth=initial_depth,
                    depth_available=depth_available,
                    scene_scale=scene_scale,
                    edge_mask=edge_mask,
                    trace_enabled=config.layerwise_trace_enabled,
                )
            xq = self.encoder(
                xq, sdpa_fn=sdpa_fn, layer_controller=layer_controller
            )
            self.last_layerwise_trace = (
                list(layer_controller.trace) if layer_controller is not None else []
            )
            q = xq[:, -q_tokens:, :]
            q = rearrange(q, "(b v) n d -> b v n d", b=batch_size, v=v2)

            # output layer
            o = self.output_layer(q)
            o = unpatchify(
                o,
                height=config.img_shape[0],
                width=config.img_shape[1],
                patch_size=config.patch_size,
            )

        return o


if __name__ == "__main__":
    # Test the model.
    import tqdm

    device = "cuda:0"
    ref_views = 2
    tar_views = 4
    batch_size = 1
    height = 256
    width = 256

    ref_imgs = torch.randn(batch_size, ref_views, height, width, 3).to(device)
    ref_cams = Camera(
        K=torch.randn(1, ref_views, 3, 3).to(device),
        camtoworld=torch.randn(1, ref_views, 4, 4).to(device),
        height=height,
        width=width,
    )
    tar_cams = Camera(
        K=torch.randn(1, tar_views, 3, 3).to(device),
        camtoworld=torch.randn(1, tar_views, 4, 4).to(device),
        height=height,
        width=width,
    )

    config = LVSMDecoderOnlyModelConfig(ref_views=2)
    model = LVSMDecoderOnlyModel(config).to(device)
    with torch.autocast("cuda"):
        for _ in tqdm.trange(100):
            y = model(ref_imgs, ref_cams, tar_cams)
        assert y.shape == (batch_size, tar_views, height, width, 3)
