"""Strict loader and harness adapter for the locked official PRoPE checkout."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch


_WORKTREE_SUFFIX = Path("worktrees/baselines/prope/workspace")
_COMMON_DIR_SUFFIX = Path("repositories/prope.git")
_LOCKED_COMMIT = "48b6dd26f1c7e906765379e4a0ef60cc50ddff7c"
_ACTIVE_BRANCH = "baseline/workspace"
_FETCH_URL = "https://github.com/liruilong940607/prope.git"
_PUSH_URL = "https://github.com/zzx135790/prope.git"
_LOCKED_SOURCE_PATHS = ("prope/__init__.py", "prope/torch.py")
_MISSING_MODULE = object()
_MODULE_CACHE_LOCK = threading.RLock()


class OfficialPropeError(RuntimeError):
    """The requested official PRoPE implementation is unavailable or unlocked."""


@dataclass(frozen=True)
class OfficialPropeProvenance:
    worktree: Path
    head: str
    branch: str
    common_dir: Path
    fetch_url: str
    push_url: str
    refs: tuple[tuple[str, str], ...]


class OfficialPropeAttentionAdapter(torch.nn.Module):
    """Drop Ray harness-only arguments before calling official PRoPE."""

    def __init__(
        self,
        official_attention: torch.nn.Module,
        *,
        official_module_file: Path,
        provenance: Optional[OfficialPropeProvenance],
    ) -> None:
        super().__init__()
        self.official_attention = official_attention
        self.official_module_file = Path(official_module_file)
        self.provenance = provenance

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        viewmats: torch.Tensor,
        Ks: Optional[torch.Tensor],
        timing_enabled: bool = False,
        predicted_d: Optional[torch.Tensor] = None,
        predicted_d_kv: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del timing_enabled, predicted_d, predicted_d_kv
        return self.official_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            **kwargs,
        )


def load_official_prope_attention(**kwargs) -> OfficialPropeAttentionAdapter:
    """Load official ``prope.torch`` only after strict checkout validation."""

    worktree, workspace_root = _canonical_worktree()
    provenance = _validate_git_provenance(worktree, workspace_root)
    locked_sources = _validate_locked_sources(worktree)
    module, attention = _fresh_load_official_attention(
        worktree, locked_sources, kwargs
    )
    module_file = Path(module.__file__)
    return OfficialPropeAttentionAdapter(
        attention,
        official_module_file=module_file,
        provenance=provenance,
    )


def _canonical_worktree(
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[Path, Path]:
    environment = os.environ if environ is None else environ
    root_value = environment.get("WORKSPACE_ROOT")
    worktree_value = environment.get("WORKSPACE_PROPE_WORKTREE")
    if not root_value and not worktree_value:
        raise OfficialPropeError(
            "Official PRoPE requires WORKSPACE_PROPE_WORKTREE or WORKSPACE_ROOT"
        )

    if root_value:
        workspace_root = _resolve_absolute_directory(root_value, "WORKSPACE_ROOT")
    else:
        candidate = _resolve_absolute_directory(
            worktree_value, "WORKSPACE_PROPE_WORKTREE"
        )
        if candidate.parts[-len(_WORKTREE_SUFFIX.parts) :] != _WORKTREE_SUFFIX.parts:
            raise OfficialPropeError(
                "WORKSPACE_PROPE_WORKTREE is not the canonical "
                f"{_WORKTREE_SUFFIX.as_posix()} path: {candidate}"
            )
        workspace_root = candidate.parents[len(_WORKTREE_SUFFIX.parts) - 1]

    marker = workspace_root / "workspace.toml"
    if not marker.is_file():
        raise OfficialPropeError(
            f"Canonical WORKSPACE_ROOT has no workspace.toml marker: {workspace_root}"
        )

    expected_worktree = (workspace_root / _WORKTREE_SUFFIX).resolve()
    if worktree_value:
        worktree = _resolve_absolute_directory(
            worktree_value, "WORKSPACE_PROPE_WORKTREE"
        )
        if worktree != expected_worktree:
            raise OfficialPropeError(
                "WORKSPACE_PROPE_WORKTREE does not match the canonical path derived "
                f"from WORKSPACE_ROOT: observed {worktree}, expected {expected_worktree}"
            )
    else:
        worktree = _resolve_absolute_directory(
            str(expected_worktree), "derived WORKSPACE_PROPE_WORKTREE"
        )
    return worktree, workspace_root


def _resolve_absolute_directory(value: Optional[str], name: str) -> Path:
    if not value:
        raise OfficialPropeError(f"{name} is empty")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise OfficialPropeError(f"{name} must be an absolute canonical path: {value!r}")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise OfficialPropeError(f"{name} does not exist: {candidate}") from error
    if not resolved.is_dir():
        raise OfficialPropeError(f"{name} is not a directory: {resolved}")
    return resolved


def _validate_git_provenance(
    worktree: Path, workspace_root: Path
) -> OfficialPropeProvenance:
    head = _git(worktree, "rev-parse", "HEAD")
    branch = _git(worktree, "symbolic-ref", "--short", "HEAD")
    common_dir_value = Path(_git(worktree, "rev-parse", "--git-common-dir"))
    common_dir = (
        common_dir_value
        if common_dir_value.is_absolute()
        else worktree / common_dir_value
    ).resolve()
    remotes = tuple(_git(worktree, "remote").splitlines())
    fetch_url = _git(worktree, "remote", "get-url", "origin")
    push_url = _git(worktree, "remote", "get-url", "--push", "origin")
    refs = tuple(
        tuple(line.split(" ", 1))
        for line in _git(
            worktree,
            "for-each-ref",
            "--format=%(refname) %(objectname)",
        ).splitlines()
    )

    expected_common_dir = (workspace_root / _COMMON_DIR_SUFFIX).resolve()
    _require_locked_ancestor(worktree, head)
    _require_git_value("branch", branch, _ACTIVE_BRANCH)
    _require_git_value("common directory", common_dir, expected_common_dir)
    _require_git_value("remotes", remotes, ("origin",))
    _require_git_value("origin fetch URL", fetch_url, _FETCH_URL)
    _require_git_value("origin push URL", push_url, _PUSH_URL)
    _require_git_value(
        "refs",
        refs,
        (
            (f"refs/heads/{_ACTIVE_BRANCH}", head),
            ("refs/remotes/origin/nvs", _LOCKED_COMMIT),
        ),
    )

    return OfficialPropeProvenance(
        worktree=worktree,
        head=head,
        branch=branch,
        common_dir=common_dir,
        fetch_url=fetch_url,
        push_url=push_url,
        refs=refs,
    )


def _require_locked_ancestor(worktree: Path, head: str) -> None:
    try:
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(worktree),
                "merge-base",
                "--is-ancestor",
                _LOCKED_COMMIT,
                head,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, UnicodeError) as error:
        raise OfficialPropeError(
            "Official PRoPE Git provenance validation could not execute git or "
            f"decode its output: {error}"
        ) from error
    if completed.returncode == 0:
        return
    if completed.returncode == 1:
        raise OfficialPropeError(
            "Official PRoPE Git provenance mismatch: active HEAD "
            f"{head} does not contain locked commit {_LOCKED_COMMIT}"
        )
    detail = completed.stderr.strip() or completed.stdout.strip()
    raise OfficialPropeError(
        "Official PRoPE Git provenance validation failed while checking locked "
        f"commit ancestry: {detail}"
    )


def _validate_locked_sources(worktree: Path) -> dict[str, bytes]:
    sources = {}
    for relative in _LOCKED_SOURCE_PATHS:
        disk_bytes = _read_regular_locked_source(worktree / relative, relative)
        locked_bytes = _git_bytes(worktree, "show", f"{_LOCKED_COMMIT}:{relative}")
        if disk_bytes != locked_bytes:
            raise OfficialPropeError(
                f"Official PRoPE locked source {relative} bytes differ from "
                f"commit {_LOCKED_COMMIT}"
            )
        sources[relative] = disk_bytes
    return sources


def _read_regular_locked_source(path: Path, relative: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise OfficialPropeError(
            f"Official PRoPE locked source {relative} is unavailable: {error}"
        ) from error
    if not stat.S_ISREG(mode):
        raise OfficialPropeError(
            f"Official PRoPE locked source {relative} must be a regular file"
        )

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OfficialPropeError(
            f"Official PRoPE locked source {relative} must be a regular file: {error}"
        ) from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OfficialPropeError(
                f"Official PRoPE locked source {relative} must be a regular file"
            )
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            return source.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fresh_load_official_attention(
    worktree: Path,
    sources: Mapping[str, bytes],
    attention_kwargs: Mapping[str, object],
):
    with _MODULE_CACHE_LOCK:
        return _fresh_load_official_attention_locked(
            worktree, sources, attention_kwargs
        )


def _fresh_load_official_attention_locked(
    worktree: Path,
    sources: Mapping[str, bytes],
    attention_kwargs: Mapping[str, object],
):
    package_path = worktree / "prope/__init__.py"
    module_path = worktree / "prope/torch.py"
    previous_package = sys.modules.get("prope", _MISSING_MODULE)
    previous_module = sys.modules.get("prope.torch", _MISSING_MODULE)

    try:
        package_spec = importlib.util.spec_from_file_location(
            "prope",
            package_path,
            submodule_search_locations=[str(package_path.parent)],
        )
        module_spec = importlib.util.spec_from_file_location(
            "prope.torch", module_path
        )
        if package_spec is None or module_spec is None:
            raise ImportError("could not create canonical module specs")

        package = importlib.util.module_from_spec(package_spec)
        sys.modules["prope"] = package
        sys.modules.pop("prope.torch", None)
        exec(
            compile(sources["prope/__init__.py"], str(package_path), "exec"),
            package.__dict__,
        )

        module = importlib.util.module_from_spec(module_spec)
        sys.modules["prope.torch"] = module
        package.torch = module
        exec(
            compile(sources["prope/torch.py"], str(module_path), "exec"),
            module.__dict__,
        )

        attention_class = getattr(module, "PropeDotProductAttention", None)
        if attention_class is None:
            raise ImportError("canonical module has no PropeDotProductAttention")
        if attention_class.__module__ != "prope.torch":
            raise ImportError(
                "canonical PropeDotProductAttention has unexpected module "
                f"{attention_class.__module__!r}"
            )
        attention = attention_class(**attention_kwargs)
        return module, attention
    except BaseException as error:
        _restore_module("prope.torch", previous_module)
        _restore_module("prope", previous_package)
        if isinstance(error, Exception):
            raise OfficialPropeError(
                "Official PRoPE fresh load of canonical prope.torch failed: "
                f"{error}"
            ) from error
        raise


def _restore_module(name: str, previous) -> None:
    if previous is _MISSING_MODULE:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = previous


def _git(worktree: Path, *arguments: str) -> str:
    command = ("git", "-C", str(worktree), *arguments)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, UnicodeError) as error:
        raise OfficialPropeError(
            "Official PRoPE Git provenance validation could not execute git or "
            f"decode its output: {error}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise OfficialPropeError(
            "Official PRoPE Git provenance validation failed for "
            f"{' '.join(arguments)}: {detail}"
        )
    return completed.stdout.strip()


def _git_bytes(worktree: Path, *arguments: str) -> bytes:
    command = ("git", "-C", str(worktree), *arguments)
    try:
        completed = subprocess.run(command, check=False, capture_output=True)
    except (OSError, UnicodeError) as error:
        raise OfficialPropeError(
            "Official PRoPE Git provenance validation could not execute git or "
            f"decode its output: {error}"
        ) from error
    if completed.returncode != 0:
        detail_bytes = completed.stderr or completed.stdout
        try:
            detail = detail_bytes.decode(errors="replace").strip()
        except UnicodeError as error:
            raise OfficialPropeError(
                "Official PRoPE Git provenance validation could not decode git "
                f"failure output: {error}"
            ) from error
        raise OfficialPropeError(
            "Official PRoPE Git provenance validation failed for "
            f"{' '.join(arguments)}: {detail}"
        )
    return completed.stdout


def _require_git_value(name, observed, expected) -> None:
    if observed != expected:
        raise OfficialPropeError(
            "Official PRoPE Git provenance mismatch for "
            f"{name}: observed {observed!r}, expected {expected!r}"
        )
