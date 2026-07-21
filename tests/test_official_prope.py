from __future__ import annotations

import builtins
import importlib
import importlib.util
import subprocess
import sys
import threading
import types
from dataclasses import fields
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from nvs.lvsm import LVSMDecoderOnlyModel, LVSMDecoderOnlyModelConfig
from pos_enc.prope import PropeDotProductAttention as VendoredPropeDotProductAttention
from pos_enc.utils.functional import Camera
from pos_enc.utils.transformer import (
    TransformerEncoderConfig,
    TransformerEncoderLayerConfig,
)


RAY_WORKTREE = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = RAY_WORKTREE.parents[3]
OFFICIAL_PROPE_WORKTREE = (
    WORKSPACE_ROOT / "worktrees/baselines/prope/workspace"
).resolve()
LOCKED_PROPE_COMMIT = "48b6dd26f1c7e906765379e4a0ef60cc50ddff7c"
ACTIVE_PROPE_COMMIT = "47163c4428c320b24db80f26d1e6faaae7ea7b50"
WORKSPACE_INTEGRATION_AVAILABLE = all(
    (
        (WORKSPACE_ROOT / "workspace.toml").is_file(),
        (OFFICIAL_PROPE_WORKTREE / ".git").exists(),
        (OFFICIAL_PROPE_WORKTREE / "prope/__init__.py").is_file(),
        (OFFICIAL_PROPE_WORKTREE / "prope/torch.py").is_file(),
    )
)
requires_workspace_prope = pytest.mark.skipif(
    not WORKSPACE_INTEGRATION_AVAILABLE,
    reason="requires canonical sibling WORKSPACE_PROPE_WORKTREE integration fixture",
)


def _official_loader_module():
    spec = importlib.util.find_spec("nvs.official_prope")
    assert spec is not None, "the official PRoPE loader/adapter is missing"
    return importlib.import_module("nvs.official_prope")


def _tiny_config(**overrides) -> LVSMDecoderOnlyModelConfig:
    layer = TransformerEncoderLayerConfig(
        d_model=16,
        nhead=1,
        dim_feedforward=32,
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
        predict_d="none",
    )
    values = {
        "ref_views": 1,
        "encoder": TransformerEncoderConfig(
            layer=layer,
            num_layers=0,
            input_norm=True,
            output_norm=True,
            checkpointing=False,
        ),
        "img_shape": (8, 8, 3),
        "cam_shape": (8, 8, 6),
        "patch_size": 8,
    }
    values.update(overrides)
    return LVSMDecoderOnlyModelConfig(**values)


def _enable_canonical_official_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(WORKSPACE_ROOT))
    monkeypatch.setenv("WORKSPACE_PROPE_WORKTREE", str(OFFICIAL_PROPE_WORKTREE))
    monkeypatch.syspath_prepend(str(OFFICIAL_PROPE_WORKTREE))
    monkeypatch.delitem(sys.modules, "prope.torch", raising=False)
    monkeypatch.delitem(sys.modules, "prope", raising=False)


def _install_cached_prope_poison(
    monkeypatch: pytest.MonkeyPatch, module_file: Path
):
    class PoisonAttention(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    PoisonAttention.__module__ = "prope.torch"
    package = types.ModuleType("prope")
    package.__file__ = str(module_file.parent / "__init__.py")
    package.__path__ = [str(module_file.parent)]
    poisoned = types.ModuleType("prope.torch")
    poisoned.__file__ = str(module_file)
    poisoned.PropeDotProductAttention = PoisonAttention
    package.torch = poisoned
    monkeypatch.setitem(sys.modules, "prope", package)
    monkeypatch.setitem(sys.modules, "prope.torch", poisoned)
    return poisoned, PoisonAttention


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(directory), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _synthetic_locked_prope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    torch_source: str | None = None,
) -> tuple[object, Path, str]:
    loader = _official_loader_module()
    workspace = tmp_path / "workspace-root"
    bare = workspace / "repositories/prope.git"
    worktree = workspace / "worktrees/baselines/prope/workspace"
    seed = tmp_path / "seed"
    bare.parent.mkdir(parents=True)
    worktree.parent.mkdir(parents=True)
    (workspace / "workspace.toml").write_text("schema_version = 3\n")
    subprocess.run(("git", "init", "--bare", str(bare)), check=True, capture_output=True)
    subprocess.run(("git", "clone", str(bare), str(seed)), check=True, capture_output=True)
    _git(seed, "config", "user.name", "Ray harness test")
    _git(seed, "config", "user.email", "ray-harness@example.invalid")
    _git(seed, "checkout", "-b", "baseline/workspace")
    package = seed / "prope"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"")
    (package / "torch.py").write_text(
        torch_source
        or (
            "import torch\n"
            "class PropeDotProductAttention(torch.nn.Module):\n"
            "    def __init__(self, **kwargs):\n"
            "        super().__init__()\n"
            "        self.kwargs = kwargs\n"
        )
    )
    _git(seed, "add", "prope/__init__.py", "prope/torch.py")
    _git(seed, "commit", "-m", "locked official core")
    locked_commit = _git(seed, "rev-parse", "HEAD")
    _git(seed, "push", "origin", "baseline/workspace")
    subprocess.run(
        (
            "git",
            "--git-dir",
            str(bare),
            "update-ref",
            "refs/remotes/origin/nvs",
            locked_commit,
        ),
        check=True,
    )
    subprocess.run(
        ("git", "--git-dir", str(bare), "remote", "add", "origin", str(bare)),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "--git-dir",
            str(bare),
            "worktree",
            "add",
            str(worktree),
            "baseline/workspace",
        ),
        check=True,
        capture_output=True,
    )
    _git(worktree, "config", "user.name", "Ray harness test")
    _git(worktree, "config", "user.email", "ray-harness@example.invalid")

    monkeypatch.setattr(loader, "_LOCKED_COMMIT", locked_commit)
    monkeypatch.setattr(loader, "_FETCH_URL", str(bare))
    monkeypatch.setattr(loader, "_PUSH_URL", str(bare))
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    monkeypatch.setenv("WORKSPACE_PROPE_WORKTREE", str(worktree))
    monkeypatch.delitem(sys.modules, "prope.torch", raising=False)
    monkeypatch.delitem(sys.modules, "prope", raising=False)
    return loader, worktree, locked_commit


def test_prope_impl_defaults_to_vendored_and_does_not_import_official(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("WORKSPACE_PROPE_WORKTREE", raising=False)
    monkeypatch.delitem(sys.modules, "prope.torch", raising=False)
    monkeypatch.delitem(sys.modules, "prope", raising=False)

    config = _tiny_config(pos_enc="prope")
    model = LVSMDecoderOnlyModel(config)

    assert config.prope_impl == "vendored"
    assert isinstance(model.attention, VendoredPropeDotProductAttention)
    assert "prope.torch" not in sys.modules


def test_prope_impl_is_last_to_preserve_existing_positional_field_order() -> None:
    assert fields(LVSMDecoderOnlyModelConfig)[-1].name == "prope_impl"


def test_official_impl_is_ignored_for_non_prope_encodings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("WORKSPACE_PROPE_WORKTREE", raising=False)

    gta_model = LVSMDecoderOnlyModel(
        _tiny_config(pos_enc="gta", prope_impl="official")
    )
    none_model = LVSMDecoderOnlyModel(
        _tiny_config(pos_enc="none", prope_impl="official")
    )

    assert isinstance(gta_model.attention, VendoredPropeDotProductAttention)
    assert none_model.attention is None


def test_prope_rejects_an_unsupported_implementation_name() -> None:
    with pytest.raises(ValueError, match=r"prope_impl.*vendored.*official"):
        LVSMDecoderOnlyModel(
            _tiny_config(pos_enc="prope", prope_impl="vendored-with-typo")
        )


def test_official_impl_requires_canonical_workspace_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _official_loader_module()
    monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("WORKSPACE_PROPE_WORKTREE", raising=False)

    with pytest.raises(
        loader.OfficialPropeError,
        match=r"WORKSPACE_PROPE_WORKTREE.*WORKSPACE_ROOT",
    ):
        LVSMDecoderOnlyModel(_tiny_config(pos_enc="prope", prope_impl="official"))


def test_official_impl_rejects_unlocked_worktree_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader = _official_loader_module()
    fake_workspace = tmp_path / "workspace-root"
    fake_workspace.mkdir()
    (fake_workspace / "workspace.toml").write_text("schema_version = 3\n")
    fake_prope = fake_workspace / "worktrees/baselines/prope/workspace"
    fake_prope.mkdir(parents=True)
    monkeypatch.setenv("WORKSPACE_ROOT", str(fake_workspace))
    monkeypatch.setenv("WORKSPACE_PROPE_WORKTREE", str(fake_prope))

    with pytest.raises(loader.OfficialPropeError, match=r"Git.*provenance"):
        LVSMDecoderOnlyModel(_tiny_config(pos_enc="prope", prope_impl="official"))


@pytest.mark.workspace_integration
@requires_workspace_prope
def test_workspace_integration_lvsm_uses_real_locked_module_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_canonical_official_import(monkeypatch)
    monkeypatch.delenv("WORKSPACE_PROPE_WORKTREE")
    loader = _official_loader_module()

    model = LVSMDecoderOnlyModel(
        _tiny_config(pos_enc="prope", prope_impl="official")
    )
    adapter = model.attention

    official_module = importlib.import_module("prope.torch")
    expected_module_file = (OFFICIAL_PROPE_WORKTREE / "prope/torch.py").resolve()
    assert isinstance(adapter, loader.OfficialPropeAttentionAdapter)
    assert Path(official_module.__file__).resolve() == expected_module_file
    assert Path(adapter.official_module_file).resolve() == expected_module_file
    assert adapter.official_attention.__class__.__module__ == "prope.torch"
    assert adapter.provenance.worktree == OFFICIAL_PROPE_WORKTREE
    assert adapter.provenance.head == ACTIVE_PROPE_COMMIT
    assert adapter.provenance.branch == "baseline/workspace"
    assert adapter.provenance.common_dir == (
        WORKSPACE_ROOT / "repositories/prope.git"
    ).resolve()
    assert adapter.provenance.fetch_url == "https://github.com/liruilong940607/prope.git"
    assert adapter.provenance.push_url == "https://github.com/zzx135790/prope.git"
    assert adapter.provenance.refs == (
        ("refs/heads/baseline/workspace", ACTIVE_PROPE_COMMIT),
        ("refs/remotes/origin/nvs", LOCKED_PROPE_COMMIT),
    )


@pytest.mark.workspace_integration
@requires_workspace_prope
def test_workspace_integration_replaces_forged_cached_official_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_canonical_official_import(monkeypatch)
    loader = _official_loader_module()
    module_file = OFFICIAL_PROPE_WORKTREE / "prope/torch.py"
    poisoned, PoisonAttention = _install_cached_prope_poison(
        monkeypatch, module_file
    )

    adapter = loader.load_official_prope_attention(
        head_dim=8,
        patches_x=2,
        patches_y=1,
        image_width=16,
        image_height=8,
    )

    assert adapter.official_attention.__class__ is not PoisonAttention
    assert sys.modules["prope.torch"] is not poisoned
    assert adapter.official_attention.__class__ is sys.modules[
        "prope.torch"
    ].PropeDotProductAttention
    assert adapter.official_attention.__class__.__module__ == "prope.torch"


def test_official_loader_rejects_modified_locked_torch_bytes_without_using_poison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader, worktree, _ = _synthetic_locked_prope(monkeypatch, tmp_path)
    module_file = worktree / "prope/torch.py"
    poisoned, _ = _install_cached_prope_poison(monkeypatch, module_file)
    module_file.write_bytes(module_file.read_bytes() + b"\n# local modification\n")

    with pytest.raises(
        loader.OfficialPropeError,
        match=r"locked source.*prope/torch.py.*bytes",
    ):
        loader.load_official_prope_attention(head_dim=8)

    assert sys.modules.get("prope.torch") is poisoned


def test_official_loader_rejects_symlinked_locked_torch_without_using_poison(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader, worktree, _ = _synthetic_locked_prope(monkeypatch, tmp_path)
    module_file = worktree / "prope/torch.py"
    replacement = tmp_path / "replacement-torch.py"
    replacement.write_bytes(module_file.read_bytes())
    module_file.unlink()
    module_file.symlink_to(replacement)
    poisoned, _ = _install_cached_prope_poison(monkeypatch, module_file)

    with pytest.raises(
        loader.OfficialPropeError,
        match=r"locked source.*prope/torch.py.*regular file",
    ):
        loader.load_official_prope_attention(head_dim=8)

    assert sys.modules.get("prope.torch") is poisoned


def test_official_loader_accepts_descendant_head_with_unchanged_locked_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader, worktree, locked_commit = _synthetic_locked_prope(monkeypatch, tmp_path)
    (worktree / "launcher.py").write_text("# downstream harness commit\n")
    _git(worktree, "add", "launcher.py")
    _git(worktree, "commit", "-m", "add downstream launcher")
    descendant_head = _git(worktree, "rev-parse", "HEAD")

    adapter = loader.load_official_prope_attention(head_dim=8)

    assert descendant_head != locked_commit
    assert adapter.provenance.head == descendant_head
    assert adapter.official_attention.__class__.__module__ == "prope.torch"


def test_official_fresh_load_failure_restores_existing_module_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader, worktree, _ = _synthetic_locked_prope(
        monkeypatch,
        tmp_path,
        torch_source="raise RuntimeError('locked import failure')\n",
    )
    poisoned, _ = _install_cached_prope_poison(
        monkeypatch, worktree / "prope/torch.py"
    )
    original_package = sys.modules["prope"]

    with pytest.raises(
        loader.OfficialPropeError,
        match=r"fresh load.*prope.torch.*locked import failure",
    ):
        loader.load_official_prope_attention(head_dim=8)

    assert sys.modules["prope"] is original_package
    assert sys.modules["prope.torch"] is poisoned
    assert original_package.torch is poisoned


def test_official_cache_transaction_is_serialized_across_failure_and_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loader, worktree, _ = _synthetic_locked_prope(
        monkeypatch,
        tmp_path,
        torch_source=(
            "import builtins\n"
            "import torch\n"
            "class PropeDotProductAttention(torch.nn.Module):\n"
            "    def __init__(self, mode, **kwargs):\n"
            "        super().__init__()\n"
            "        builtins._ray_prope_concurrency_hook(mode)\n"
        ),
    )
    poisoned, _ = _install_cached_prope_poison(
        monkeypatch, worktree / "prope/torch.py"
    )
    failure_inside = threading.Event()
    success_inside = threading.Event()
    allow_failure = threading.Event()
    allow_success = threading.Event()
    failure_finished = threading.Event()
    success_finished = threading.Event()
    results = {}
    errors = {}

    def concurrency_hook(mode):
        if mode == "fail":
            failure_inside.set()
            assert allow_failure.wait(5), "failure thread release timed out"
            raise RuntimeError("intentional concurrent load failure")
        success_inside.set()
        assert allow_success.wait(5), "success thread release timed out"

    monkeypatch.setattr(
        builtins, "_ray_prope_concurrency_hook", concurrency_hook, raising=False
    )

    def load(mode, finished):
        try:
            results[mode] = loader.load_official_prope_attention(
                head_dim=8, mode=mode
            )
        except BaseException as error:
            errors[mode] = error
        finally:
            finished.set()

    failure_thread = threading.Thread(
        target=load, args=("fail", failure_finished), daemon=True
    )
    success_thread = threading.Thread(
        target=load, args=("success", success_finished), daemon=True
    )
    failure_thread.start()
    assert failure_inside.wait(5), "failure thread never entered module construction"
    success_thread.start()

    transactions_overlapped = success_inside.wait(0.5)
    if transactions_overlapped:
        allow_success.set()
        assert success_finished.wait(5), "overlapped success did not finish"
        allow_failure.set()
    else:
        allow_failure.set()
        assert failure_finished.wait(5), "serialized failure did not finish"
        assert success_inside.wait(5), "serialized success never acquired transaction"
        allow_success.set()

    failure_thread.join(5)
    success_thread.join(5)
    assert not failure_thread.is_alive()
    assert not success_thread.is_alive()
    assert isinstance(errors.get("fail"), loader.OfficialPropeError)
    assert "success" in results
    final_module = sys.modules["prope.torch"]
    assert final_module is not poisoned
    assert results["success"].official_attention.__class__ is (
        final_module.PropeDotProductAttention
    )


def test_git_text_helper_wraps_unicode_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _official_loader_module()
    decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")

    def fail_decode(*args, **kwargs):
        raise decode_error

    monkeypatch.setattr(loader.subprocess, "run", fail_decode)

    with pytest.raises(loader.OfficialPropeError, match=r"execute git.*decode"):
        loader._git(Path("/synthetic/worktree"), "status")


def test_git_bytes_failure_uses_stdout_when_stderr_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader = _official_loader_module()
    completed = types.SimpleNamespace(
        returncode=1,
        stderr=b"",
        stdout=b"locked blob missing from stdout",
    )
    monkeypatch.setattr(loader.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(
        loader.OfficialPropeError,
        match=r"locked blob missing from stdout",
    ):
        loader._git_bytes(Path("/synthetic/worktree"), "show", "locked:path")


def test_official_adapter_drops_harness_only_arguments() -> None:
    loader = _official_loader_module()
    calls = []

    class RecordingAttention(torch.nn.Module):
        def forward(self, q, k, v, viewmats, Ks, **kwargs):
            calls.append((q, k, v, viewmats, Ks, kwargs))
            return q + v

    recorder = RecordingAttention()
    adapter = loader.OfficialPropeAttentionAdapter(
        recorder,
        official_module_file=Path("/canonical/prope/torch.py"),
        provenance=None,
    )
    q = torch.ones(1, 1, 1, 1)
    k = torch.full_like(q, 2.0)
    v = torch.full_like(q, 3.0)
    viewmats = torch.eye(4).reshape(1, 1, 4, 4)
    Ks = torch.eye(3).reshape(1, 1, 3, 3)

    output = adapter(
        q,
        k,
        v,
        viewmats=viewmats,
        Ks=Ks,
        timing_enabled=True,
        predicted_d=torch.zeros(1, 1, 2),
        predicted_d_kv=None,
        dropout_p=0.0,
        is_causal=False,
    )

    assert torch.equal(output, q + v)
    assert calls == [(q, k, v, viewmats, Ks, {"dropout_p": 0.0, "is_causal": False})]


@pytest.mark.workspace_integration
@requires_workspace_prope
def test_workspace_integration_official_and_vendored_are_bit_identical_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_canonical_official_import(monkeypatch)
    loader = _official_loader_module()
    import pos_enc.prope as vendored_module

    for helper_name in (
        "_prepare_apply_fns",
        "_apply_tiled_projmat",
        "_rope_precompute_coeffs",
        "_rope_apply_coeffs",
    ):
        compiled_helper = getattr(vendored_module, helper_name)
        eager_helper = getattr(
            compiled_helper, "_torchdynamo_orig_callable", compiled_helper
        )
        monkeypatch.setattr(vendored_module, helper_name, eager_helper)

    kwargs = {
        "head_dim": 8,
        "patches_x": 2,
        "patches_y": 1,
        "image_width": 16,
        "image_height": 8,
    }
    vendored = VendoredPropeDotProductAttention(**kwargs)
    official = loader.load_official_prope_attention(**kwargs)

    generator = torch.Generator(device="cpu").manual_seed(1729)
    q = torch.randn(1, 2, 4, 8, generator=generator)
    k = torch.randn(1, 2, 4, 8, generator=generator)
    v = torch.randn(1, 2, 4, 8, generator=generator)
    viewmats = torch.eye(4).repeat(1, 2, 1, 1)
    viewmats[:, 1, 0, 3] = 0.25
    Ks = torch.tensor(
        [[[[12.0, 0.0, 8.0], [0.0, 11.0, 4.0], [0.0, 0.0, 1.0]],
          [[10.0, 0.0, 7.0], [0.0, 13.0, 5.0], [0.0, 0.0, 1.0]]]],
    )

    vendored_output = vendored(
        q,
        k,
        v,
        viewmats=viewmats,
        Ks=Ks,
        timing_enabled=False,
        dropout_p=0.0,
        is_causal=False,
    )
    official_output = official(
        q,
        k,
        v,
        viewmats=viewmats,
        Ks=Ks,
        timing_enabled=True,
        dropout_p=0.0,
        is_causal=False,
    )

    assert torch.equal(official_output, vendored_output)


def test_camray_uses_identity_c2w_then_converts_raymap_to_plucker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nvs.lvsm as lvsm

    model = LVSMDecoderOnlyModel(
        _tiny_config(pos_enc="none", ray_encoding="camray")
    )
    original_c2w = torch.eye(4, dtype=torch.float64).repeat(1, 2, 1, 1)
    original_c2w[:, 0, 0, 3] = 3.0
    original_c2w[:, 1, 1, 3] = -2.0
    cameras = Camera(
        K=torch.eye(3, dtype=torch.float64).repeat(1, 2, 1, 1),
        camtoworld=original_c2w,
        width=8,
        height=8,
    )
    raymap = torch.randn(1, 2, 8, 8, 3, dtype=torch.float64)
    plucker = torch.randn(1, 2, 8, 8, 6, dtype=torch.float64)
    observed = {}

    def fake_camera_to_raymap(**kwargs):
        observed.update(kwargs)
        return raymap

    def fake_raymap_to_plucker(value):
        assert value is raymap
        return plucker

    monkeypatch.setattr(lvsm, "camera_to_raymap", fake_camera_to_raymap)
    monkeypatch.setattr(lvsm, "raymap_to_plucker", fake_raymap_to_plucker)

    result = model.create_rays(cameras)

    expected_identity = torch.eye(4, dtype=torch.float64).broadcast_to(
        original_c2w.shape
    )
    assert result is plucker
    assert observed["Ks"] is cameras.K
    assert torch.equal(observed["camtoworlds"], expected_identity)
    assert observed["camtoworlds"].dtype == original_c2w.dtype
    assert observed["camtoworlds"].device == original_c2w.device
    assert observed["height"] == 8
    assert observed["width"] == 8
    assert observed["downscale"] == 1
