"""RayRoPE provider adapter for the shared ``rope-contract`` middleware.

RayRoPE's native callable performs both positional preparation and SDPA.  The
provider adapter deliberately calls only its precomputation/apply functions:
the consumer owns SDPA and supplies one attention message per query camera to
``restore_output``.  The unchanged native module remains available through
``legacy_native`` as an explicit rollback path.

The baseline implementation currently exposes the documented self-attention
path.  The released cross helper has a known KV-camera reshape defect; cross
requests fail closed with ``UnsupportedCapability`` rather than silently
falling back to a different geometry convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

try:
    from rope_contract import (
        CANONICAL_LAYOUT_ID,
        OP_TRANSFORM_INPUTS,
        CapabilityManifest,
        OpaqueContinuation,
        PrepareRequest,
        ProviderProfile,
        RopeProvider,
        RopeSession,
        SemanticChannel,
        ShapeMismatch,
        TensorDescriptor,
        TransformRequest,
        TransformResult,
        UnsupportedCapability,
        InvalidContinuation,
        ValidationError,
    )
except ImportError as exc:  # pragma: no cover - depends on workspace env
    raise ImportError(
        "RayRoPE contract integration requires rope-contract on PYTHONPATH"
    ) from exc

from pos_enc.rayrope import RayRoPE_DotProductAttention


PROVIDER_ID = "rayrope"
ADAPTER_ID = "rayrope-contract-provider"
ADAPTER_VERSION = "1"
TOKEN_ORDER = "camera_major_token_major"


def _tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ShapeMismatch(f"{name} must be a torch.Tensor", actual=type(value).__name__)
    if not value.is_floating_point():
        raise ValidationError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValidationError(f"{name} must contain finite values")
    return value


def _descriptor(value: torch.Tensor, *, role: str, token_order: str = TOKEN_ORDER) -> TensorDescriptor:
    return TensorDescriptor.from_tensor(
        value, layout_id=CANONICAL_LAYOUT_ID, role=role, token_order=token_order
    )


def _same(left: Any, right: Any) -> bool:
    return left is right or (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(left, right)
    )


def _select(payload: Mapping[str, Any], aliases: tuple[str, ...], *, name: str) -> Any:
    found = [(key, payload[key]) for key in aliases if key in payload]
    if not found:
        return None
    value = found[0][1]
    if any(not _same(value, candidate) for _, candidate in found[1:]):
        raise ValidationError(
            f"conflicting aliases were supplied for {name}",
            actual=[key for key, _ in found],
        )
    return value


@dataclass(frozen=True)
class RayRoPEGeometry:
    """Geometry retained by a RayRoPE session.

    ``predicted_d`` follows the native ``[B, C*P, 2]`` log-depth/log-width
    convention.  It is carried in the transform request because it may be
    predicted by the benchmark immediately before attention.
    """

    w2cs: torch.Tensor
    intrinsics: torch.Tensor
    context_depths: Optional[torch.Tensor] = None
    depth_source: str = "predicted"
    batch_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w2cs = _tensor(self.w2cs, "w2cs")
        intrinsics = _tensor(self.intrinsics, "intrinsics")
        if w2cs.ndim != 4 or tuple(w2cs.shape[-2:]) != (4, 4):
            raise ShapeMismatch("w2cs must have shape [B,C,4,4]", actual=tuple(w2cs.shape))
        if intrinsics.ndim != 4 or tuple(intrinsics.shape[-2:]) != (3, 3):
            raise ShapeMismatch(
                "intrinsics must have shape [B,C,3,3]", actual=tuple(intrinsics.shape)
            )
        if w2cs.shape[:2] != intrinsics.shape[:2]:
            raise ShapeMismatch("w2cs and intrinsics must share [B,C]")
        if w2cs.shape[0] <= 0 or w2cs.shape[1] <= 0:
            raise ShapeMismatch("w2cs must contain at least one batch and camera")
        if w2cs.device != intrinsics.device or w2cs.dtype != intrinsics.dtype:
            raise ValidationError("w2cs and intrinsics must share dtype and device")
        object.__setattr__(self, "w2cs", w2cs)
        object.__setattr__(self, "intrinsics", intrinsics)
        if self.context_depths is not None:
            depths = _tensor(self.context_depths, "context_depths")
            if depths.ndim != 5 or depths.shape[:2] != w2cs.shape[:2] or depths.shape[-1] != 1:
                raise ShapeMismatch(
                    "context_depths must have shape [B,C,H,W,1]",
                    actual=tuple(depths.shape),
                )
            if depths.device != w2cs.device:
                raise ValidationError("context_depths must share geometry device")
            object.__setattr__(self, "context_depths", depths)
        source = str(self.depth_source).strip().lower()
        if source not in {"predicted", "known+predicted"}:
            raise ValidationError(
                "RayRoPE depth_source must be predicted or known+predicted", actual=source
            )
        object.__setattr__(self, "depth_source", source)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def batch_size(self) -> int:
        return int(self.w2cs.shape[0])

    @property
    def camera_count(self) -> int:
        return int(self.w2cs.shape[1])

    def payload(self) -> dict[str, Any]:
        return {
            "w2cs": self.w2cs,
            "intrinsics": self.intrinsics,
            "context_depths": self.context_depths,
            "depth_source": self.depth_source,
            "geometry_metadata": dict(self.metadata),
        }


def _channels(geometry: RayRoPEGeometry) -> dict[str, SemanticChannel]:
    dtype = str(geometry.w2cs.dtype).removeprefix("torch.")
    return {
        "w2c": SemanticChannel(
            "w2c", shape=tuple(geometry.w2cs.shape), frame="world_to_camera",
            units="scene_units", ownership="camera", alignment="camera_major",
            dtype=dtype, source="geometry", physical_length=6,
        ),
        "intrinsics": SemanticChannel(
            "intrinsics", shape=tuple(geometry.intrinsics.shape), frame="camera",
            units="pixel", ownership="camera", alignment="camera_major",
            dtype=dtype, source="geometry", physical_length=4,
        ),
        "depth": SemanticChannel(
            "depth", frame="camera", units="log_depth", ownership="token",
            alignment=TOKEN_ORDER, dtype="float32", source=geometry.depth_source,
            available=geometry.depth_source in {"predicted", "known+predicted"},
            physical_length=1,
        ),
    }


def make_prepare_request(
    geometry: RayRoPEGeometry,
    *,
    patches_x: int,
    patches_y: int,
    profile_id: str = "rayrope_v1",
    provider_id: str = PROVIDER_ID,
    consumer_id: Optional[str] = None,
    consumer_version: Optional[str] = None,
) -> PrepareRequest:
    """Build a typed preparation envelope from native RayRoPE geometry."""

    if isinstance(patches_x, bool) or not isinstance(patches_x, int) or patches_x <= 0:
        raise ShapeMismatch("patches_x must be a positive integer", actual=patches_x)
    if isinstance(patches_y, bool) or not isinstance(patches_y, int) or patches_y <= 0:
        raise ShapeMismatch("patches_y must be a positive integer", actual=patches_y)
    token_count = geometry.camera_count * patches_x * patches_y
    metadata = {**dict(geometry.metadata), "patches_x": patches_x, "patches_y": patches_y,
                "token_count": token_count}
    descriptors = {
        "w2cs": _descriptor(geometry.w2cs, role="attention_message", token_order="camera_major"),
        "intrinsics": _descriptor(geometry.intrinsics, role="attention_message", token_order="camera_major"),
    }
    return PrepareRequest(
        profile_id=profile_id, provider_id=provider_id, consumer_id=consumer_id,
        consumer_version=consumer_version, attention_kind="multi_camera_dense_self",
        operation=OP_TRANSFORM_INPUTS, layout_id=CANONICAL_LAYOUT_ID,
        tensor_descriptors=descriptors, semantic_channels=_channels(geometry),
        logical_shape=tuple(geometry.w2cs.shape), token_order=TOKEN_ORDER,
        batch_id=geometry.batch_id, depth_source=geometry.depth_source,
        payload=geometry.payload(), metadata=metadata,
    )


def make_transform_request(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    profile_id: str,
    predicted_d: Optional[torch.Tensor] = None,
    continuation: Optional[OpaqueContinuation] = None,
    dropout_p: float = 0.0,
    training: bool = False,
    batch_id: Optional[str] = None,
    session_generation: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> TransformRequest:
    q, k, v = (_tensor(query, "query"), _tensor(key, "key"), _tensor(value, "value"))
    payload = {"predicted_d": predicted_d}
    return TransformRequest(
        profile_id=profile_id, attention_kind="multi_camera_dense_self",
        operation=OP_TRANSFORM_INPUTS, query=q, key=k, value=v,
        tensor_descriptors={
            "query": _descriptor(q, role="query"),
            "key": _descriptor(k, role="key"),
            "value": _descriptor(v, role="value"),
        }, continuation=continuation, dropout_p=dropout_p, training=training,
        batch_id=batch_id, session_generation=session_generation,
        # Tensor payloads stay in ``operands``; continuation hashing serializes
        # metadata and must never attempt to encode a framework tensor.
        metadata=dict(metadata or {}),
        operands=payload,
    )


def _profile(*, head_dim: int, apply_vo: bool, config: Mapping[str, Any]) -> ProviderProfile:
    return ProviderProfile(
        profile_id="rayrope_v1", profile_version="1", feature_id="rayrope_v1",
        family="ray", strategy="deterministic", operations=("transform_inputs", "restore_output"),
        attention_kinds=("self", "multi_camera_dense_self"),
        transformed_roles=("query", "key", "value"), head_roles=("ray", "content"),
        required_semantic_channels=("w2c", "intrinsics", "depth"),
        head_dim_values=(head_dim,), dtype_values=("float16", "float32", "float64", "bfloat16"),
        device_values=("cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"),
        supports_gradients=True, supports_masks=False, supports_causal=False,
        supports_dropout=True, stateless=False, session_isolation=True, rng_isolation=True,
        metadata={
            "execution_mode": "transformed_operands",
            "attention_kernel_owner": "consumer",
            "native_callable": "RayRoPE_DotProductAttention",
            "value_policy": "native_apply_vo" if apply_vo else "pass_through",
            "tail_policy": "first_120_transformed_plus_8_neutral",
            "geometry_convention": "world_to_camera+pixel_intrinsics",
            "token_order": TOKEN_ORDER,
            "config": dict(config),
        },
    )


class RayRoPESession(RopeSession):
    def __init__(self, *, provider: "RayRoPEProvider", profile: ProviderProfile,
                 prepare_request: PrepareRequest, geometry: RayRoPEGeometry) -> None:
        super().__init__(provider=provider, manifest=provider.manifest, profile=profile,
                         prepare_request=prepare_request)
        self.provider = provider
        self.geometry = geometry
        self._pending: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, int, int]] = {}
        self._native: Optional[RayRoPE_DotProductAttention] = None

    def close(self) -> None:
        self._pending.clear()
        self._native = None
        super().close()

    def _transform_inputs_impl(self, request: TransformRequest) -> TransformResult:
        if request.attention_kind not in {"self", "multi_camera_dense_self"}:
            raise UnsupportedCapability("RayRoPE baseline provider supports self attention only", actual=request.attention_kind)
        q, k, v = request.query, request.key, request.value
        assert isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)
        if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
            raise ShapeMismatch("RayRoPE expects Q/K/V [B,H,C*P,D] with equal shapes")
        b, _, tokens, head_dim = q.shape
        if b != self.geometry.batch_size or q.device != self.geometry.w2cs.device:
            raise ShapeMismatch("Q/K/V batch or device differs from prepared geometry")
        patches_x = int(self.prepare_request.metadata["patches_x"])
        patches_y = int(self.prepare_request.metadata["patches_y"])
        patches = patches_x * patches_y
        expected_tokens = self.geometry.camera_count * patches
        if tokens != expected_tokens:
            raise ShapeMismatch("token count does not match camera grid", expected=expected_tokens, actual=tokens)
        predicted_d = _select(request.operands, ("predicted_d",), name="predicted_d")
        if predicted_d is None:
            predicted_d = request.metadata.get("predicted_d")
        predicted_d = _tensor(predicted_d, "predicted_d") if predicted_d is not None else None
        if predicted_d is None:
            raise ValidationError("RayRoPE requires predicted_d [B,C*P,2] for its native depth encoding")
        if predicted_d.shape != (b, tokens, 2) or predicted_d.device != q.device:
            raise ShapeMismatch("predicted_d must have shape [B,C*P,2] on the Q/K/V device", actual=tuple(predicted_d.shape))
        # The published RayRoPE geometry consumes the first 120 channels of
        # the canonical 128-channel head.  The final eight channels are a
        # neutral, exact identity tail retained by this adapter.
        transformed_dim = 120 if head_dim == 128 else head_dim
        if self.provider.depth_type == "known+predict_dsig" and self.geometry.context_depths is None:
            raise ValidationError(
                "known+predict_dsig requires context_depths [B,C,H,W,1]"
            )
        if self.provider.depth_type == "predict_dsig" and self.geometry.context_depths is not None:
            raise ValidationError(
                "predict_dsig does not accept context_depths; select known+predict_dsig explicitly"
            )
        native = RayRoPE_DotProductAttention(
            head_dim=transformed_dim, patches_x=patches_x, patches_y=patches_y,
            image_width=int(self.provider.image_width), image_height=int(self.provider.image_height),
            pos_enc_type=self.provider.pos_enc_type, num_rays_per_patch=self.provider.num_rays_per_patch,
            depth_type=self.provider.depth_type,
            denc_type=self.provider.denc_type, freq_base=self.provider.freq_base,
            apply_vo=self.provider.apply_vo,
        ).to(device=q.device)
        native._precompute_and_cache_apply_fns(self.geometry.w2cs, self.geometry.intrinsics, self.geometry.context_depths)
        wrapped = getattr(native._prepare_apply_fns, "__wrapped__", None)
        fns = wrapped(native, predicted_d=predicted_d) if wrapped is not None else native._prepare_apply_fns(predicted_d=predicted_d)
        q_fn, kv_fns, out_fn = fns
        q_encoded = q_fn(q[..., :transformed_dim])
        q_heads = torch.cat((q_encoded, q[..., transformed_dim:]), dim=-1)
        q_heads = q_heads.reshape(b, q_heads.shape[1], native.num_cameras, patches, head_dim)
        # Move camera into the logical batch expected by the consumer SDPA.
        q_heads = q_heads.permute(0, 2, 1, 3, 4).reshape(b * native.num_cameras, q_heads.shape[1], patches, head_dim)
        k_per_camera = [torch.cat((fn(k[..., :transformed_dim]), k[..., transformed_dim:]), dim=-1) for fn in kv_fns]
        v_per_camera = [
            torch.cat((fn(v[..., :transformed_dim]), v[..., transformed_dim:]), dim=-1)
            if self.provider.apply_vo else v
            for fn in kv_fns
        ]
        k_heads = torch.stack(k_per_camera, dim=1).reshape(b * native.num_cameras, k.shape[1], tokens, head_dim)
        v_heads = torch.stack(v_per_camera, dim=1).reshape(b * native.num_cameras, v.shape[1], tokens, head_dim)
        continuation = self._issue_continuation(operation=OP_TRANSFORM_INPUTS, request=request, output_layout_id=CANONICAL_LAYOUT_ID)
        self._pending[continuation.token] = (q_heads, k_heads, v_heads, out_fn, b, native.num_cameras)
        descriptors = {
            "query": _descriptor(q_heads, role="query"),
            "key": _descriptor(k_heads, role="key"),
            "value": _descriptor(v_heads, role="value"),
        }
        self._native = native
        return TransformResult(query=q_heads, key=k_heads, value=v_heads, continuation=continuation,
                               tensor_descriptors=descriptors,
                               metadata={"execution_mode": "transformed_operands", "consumer_attention": "scaled_dot_product_attention",
                                         "query_shape": list(q_heads.shape), "message_shape": [b * native.num_cameras, q.shape[1], patches, head_dim],
                                         "value_policy": "native_apply_vo" if self.provider.apply_vo else "pass_through"})

    def _restore_output_impl(self, attention_message: Any, *, continuation: OpaqueContinuation, request: Any = None) -> TransformResult:
        del request
        try:
            q, k, v, out_fn, batch, cameras = self._pending[continuation.token]
        except KeyError as exc:
            raise InvalidContinuation("unknown or already restored RayRoPE continuation") from exc
        message = _tensor(attention_message, "attention_message")
        expected = q.shape
        if tuple(message.shape) != tuple(expected) or message.device != q.device or message.dtype != q.dtype:
            raise ShapeMismatch("attention message must match transformed query", expected=tuple(expected), actual=tuple(message.shape))
        logical_message = message.reshape(batch, cameras, message.shape[1], message.shape[2], message.shape[3]).permute(0, 2, 1, 3, 4).reshape(batch, message.shape[1], cameras * message.shape[2], message.shape[3])
        transformed_dim = 120 if logical_message.shape[-1] == 128 else logical_message.shape[-1]
        if self.provider.apply_vo:
            restored = torch.cat(
                (out_fn(logical_message[..., :transformed_dim]), logical_message[..., transformed_dim:]),
                dim=-1,
            )
        else:
            # ``apply_vo=False`` is the native Q/K-only mode: values and the
            # attention message remain in the consumer's original basis.
            restored = logical_message
        del self._pending[continuation.token]
        return TransformResult(output=restored, tensor_descriptors={"output": _descriptor(restored, role="output")},
                               metadata={"execution_mode": "restored_output", "output_shape": list(restored.shape)})


class RayRoPEProvider(RopeProvider):
    """Provider façade around the unmodified RayRoPE callable implementation."""

    def __init__(self, *, patches_x: int, patches_y: int, image_width: int, image_height: int,
                 head_dim: int = 128, pos_enc_type: str = "d_pj+0_3d",
                 num_rays_per_patch: int = 3, depth_type: str = "predict_dsig",
                 denc_type: str = "d", freq_base: float = 3.0, apply_vo: bool = True,
                 provider_id: str = PROVIDER_ID) -> None:
        if head_dim <= 0:
            raise ShapeMismatch("head_dim must be positive", actual=head_dim)
        if patches_x <= 0 or patches_y <= 0 or image_width <= 0 or image_height <= 0:
            raise ShapeMismatch("patch grid and image dimensions must be positive")
        if depth_type not in {"predict_dsig", "known+predict_dsig"}:
            raise UnsupportedCapability("unsupported RayRoPE depth_type", actual=depth_type)
        self.patches_x, self.patches_y = int(patches_x), int(patches_y)
        self.image_width, self.image_height = int(image_width), int(image_height)
        self.head_dim = int(head_dim)
        if self.head_dim != 128:
            raise UnsupportedCapability(
                "RayRoPE contract profile is fixed to the documented 120+8 head layout",
                actual=self.head_dim,
            )
        self.pos_enc_type = str(pos_enc_type)
        self.num_rays_per_patch = int(num_rays_per_patch)
        self.depth_type, self.denc_type = depth_type, str(denc_type)
        self.freq_base, self.apply_vo = float(freq_base), bool(apply_vo)
        config = {"head_dim": self.head_dim, "patches_x": self.patches_x, "patches_y": self.patches_y,
                  "image_width": self.image_width, "image_height": self.image_height,
                  "pos_enc_type": self.pos_enc_type, "num_rays_per_patch": self.num_rays_per_patch,
                  "depth_type": self.depth_type, "denc_type": self.denc_type,
                  "freq_base": self.freq_base, "apply_vo": self.apply_vo}
        profile = _profile(head_dim=self.head_dim, apply_vo=self.apply_vo, config=config)
        manifest = CapabilityManifest(provider_id=provider_id, provider_version="rayrope-main",
                                       profiles=(profile,), adapter_id=ADAPTER_ID,
                                       adapter_version=ADAPTER_VERSION,
                                       metadata={"native_entrypoint": "pos_enc.rayrope.RayRoPE_DotProductAttention"})
        super().__init__(manifest=manifest)

    def _geometry(self, request: PrepareRequest) -> RayRoPEGeometry:
        payload = request.payload
        w2cs = _select(payload, ("w2cs", "w2c"), name="w2cs")
        intrinsics = _select(payload, ("intrinsics", "Ks", "K"), name="intrinsics")
        if w2cs is None or intrinsics is None:
            raise ValidationError("PrepareRequest payload requires w2cs and intrinsics")
        w2cs, intrinsics = _tensor(w2cs, "w2cs"), _tensor(intrinsics, "intrinsics")
        for role, tensor in (("w2cs", w2cs), ("intrinsics", intrinsics)):
            descriptor = request.tensor_descriptors.get(role)
            if descriptor is not None:
                descriptor.validate(expected_shape=tuple(tensor.shape), expected_layout=CANONICAL_LAYOUT_ID,
                                    expected_dtype=str(tensor.dtype), expected_device=str(tensor.device),
                                    expected_role="attention_message", require_token_order=True)
        return RayRoPEGeometry(
            w2cs=w2cs, intrinsics=intrinsics,
            context_depths=_select(payload, ("context_depths", "depths"), name="context_depths"),
            depth_source=request.depth_source or str(payload.get("depth_source", "predicted")),
            batch_id=request.batch_id,
            metadata={**dict(payload.get("geometry_metadata", {})), **dict(request.metadata)},
        )

    def _create_session(self, request: PrepareRequest, profile: ProviderProfile) -> RopeSession:
        geometry = self._geometry(request)
        if int(request.metadata.get("patches_x", self.patches_x)) != self.patches_x or int(request.metadata.get("patches_y", self.patches_y)) != self.patches_y:
            raise ShapeMismatch("prepare patch grid differs from provider configuration")
        return RayRoPESession(provider=self, profile=profile, prepare_request=request, geometry=geometry)

    def open_session_for_geometry(self, geometry: RayRoPEGeometry, *, profile_id: str = "rayrope_v1",
                                  consumer_id: Optional[str] = None, consumer_version: Optional[str] = None) -> RayRoPESession:
        request = make_prepare_request(geometry, patches_x=self.patches_x, patches_y=self.patches_y,
                                       profile_id=profile_id, provider_id=self.provider_id,
                                       consumer_id=consumer_id, consumer_version=consumer_version)
        return self.open_session(request)  # type: ignore[return-value]

    def legacy_native(self, *, device: Optional[torch.device] = None) -> RayRoPE_DotProductAttention:
        """Construct the original callable attention module for rollback."""
        native = RayRoPE_DotProductAttention(
            120, patches_x=self.patches_x, patches_y=self.patches_y,
            image_width=self.image_width, image_height=self.image_height,
            pos_enc_type=self.pos_enc_type, num_rays_per_patch=self.num_rays_per_patch,
            depth_type=self.depth_type, denc_type=self.denc_type,
            freq_base=self.freq_base, apply_vo=self.apply_vo,
        )
        return native if device is None else native.to(device=device)


__all__ = [
    "RayRoPEGeometry", "RayRoPEProvider", "RayRoPESession",
    "make_prepare_request", "make_transform_request",
]
