"""Focused contract tests for the RayRoPE provider boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[5]
CONTRACT_SRC = ROOT / "worktrees" / "rope-contract" / "mainline" / "src"
if CONTRACT_SRC.is_dir():
    sys.path.insert(0, str(CONTRACT_SRC))

from pos_enc.integrations.rope_contract_provider import (  # noqa: E402
    RayRoPEGeometry,
    RayRoPEProvider,
    make_transform_request,
)
from rope_contract import UnsupportedCapability, ValidationError  # noqa: E402


def _geometry() -> RayRoPEGeometry:
    w2c = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    w2c[:, 1, 0, 3] = 0.25
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)
    intrinsics[..., 0, 0] = intrinsics[..., 1, 1] = 2.0
    return RayRoPEGeometry(w2c, intrinsics)


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(5)
    return tuple(torch.randn(1, 1, 8, 128) for _ in range(3))  # type: ignore[return-value]


def test_manifest_and_legacy_rollback_are_explicit() -> None:
    provider = RayRoPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    profile = provider.describe().profile("rayrope_v1")
    assert profile.metadata["attention_kernel_owner"] == "consumer"
    assert profile.metadata["value_policy"] == "native_apply_vo"
    assert provider.legacy_native().head_dim == 120


def test_transform_prepares_per_camera_operands_without_running_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RayRoPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    session = provider.open_session_for_geometry(_geometry())
    q, k, v = _qkv()
    predicted = torch.zeros(1, 8, 2)

    def fail(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("RayRoPE provider must not execute consumer SDPA")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fail)
    prepared = session.transform_inputs(
        make_transform_request(
            q, k, v, profile_id="rayrope_v1", predicted_d=predicted,
            session_generation=session.session_generation,
        )
    )
    assert prepared.output is None
    assert prepared.query.shape == (2, 1, 4, 128)
    assert prepared.key.shape == prepared.value.shape == (2, 1, 8, 128)
    # The documented 8-channel tail is untouched by geometry preparation.
    torch.testing.assert_close(prepared.value[..., 120:], v[..., 120:].expand(2, -1, -1, -1))


def test_consumer_message_restores_to_original_layout() -> None:
    provider = RayRoPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    session = provider.open_session_for_geometry(_geometry())
    q, k, v = _qkv()
    prepared = session.transform_inputs(
        make_transform_request(q, k, v, profile_id="rayrope_v1", predicted_d=torch.zeros(1, 8, 2))
    )
    message = torch.nn.functional.scaled_dot_product_attention(prepared.query, prepared.key, prepared.value)
    restored = session.restore_output(message, continuation=prepared.continuation)
    assert restored.output is not None and restored.output.shape == q.shape
    assert torch.isfinite(restored.output).all()


def test_zero_tail_matches_legacy_callable_attention() -> None:
    """With the neutral tail disabled, provider and native 120-D paths agree."""
    provider = RayRoPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    geometry = _geometry()
    q, k, v = _qkv()
    q[..., 120:] = 0
    k[..., 120:] = 0
    v[..., 120:] = 0
    predicted = torch.zeros(1, 8, 2)
    native = provider.legacy_native()
    native._precompute_and_cache_apply_fns(geometry.w2cs, geometry.intrinsics)
    expected = native(q[..., :120], k[..., :120], v[..., :120], predicted_d=predicted)
    session = provider.open_session_for_geometry(geometry)
    prepared = session.transform_inputs(
        make_transform_request(q, k, v, profile_id="rayrope_v1", predicted_d=predicted)
    )
    message = torch.nn.functional.scaled_dot_product_attention(prepared.query, prepared.key, prepared.value)
    restored = session.restore_output(message, continuation=prepared.continuation).output
    torch.testing.assert_close(restored[..., :120], expected, rtol=2e-4, atol=2e-4)


def test_cross_attention_and_missing_depth_fail_closed() -> None:
    provider = RayRoPEProvider(patches_x=2, patches_y=2, image_width=2, image_height=2)
    session = provider.open_session_for_geometry(_geometry())
    q, k, v = _qkv()
    request = make_transform_request(q, k, v, profile_id="rayrope_v1", predicted_d=torch.zeros(1, 8, 2))
    request = type(request)(**{**request.__dict__, "attention_kind": "cross"})
    with pytest.raises(UnsupportedCapability):
        session.transform_inputs(request)
    with pytest.raises(ValidationError):
        session.transform_inputs(make_transform_request(q, k, v, profile_id="rayrope_v1"))
