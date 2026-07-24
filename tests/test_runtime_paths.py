from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest


def test_objaverse_renderer_has_no_machine_specific_grogu_paths() -> None:
    renderer = Path(__file__).resolve().parents[1] / "scripts/objv_render_vary_intrinsics.py"

    assert "/grogu/" not in renderer.read_text(encoding="utf-8")


def test_objaverse_renderer_requires_an_explicit_output_directory() -> None:
    renderer = Path(__file__).resolve().parents[1] / "scripts/objv_render_vary_intrinsics.py"
    tree = ast.parse(renderer.read_text(encoding="utf-8"))
    output_argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--output_dir"
    )

    assert any(
        keyword.arg == "required"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in output_argument.keywords
    )
    assert all(keyword.arg != "default" for keyword in output_argument.keywords)


def test_objaverse_renderer_requires_an_explicit_log_file() -> None:
    renderer = Path(__file__).resolve().parents[1] / "scripts/objv_render_vary_intrinsics.py"
    tree = ast.parse(renderer.read_text(encoding="utf-8"))
    log_argument = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--dump_log"
    )

    assert any(
        keyword.arg == "required"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in log_argument.keywords
    )
    assert all(keyword.arg != "default" for keyword in log_argument.keywords)


def test_objaverse_submitit_task_passes_an_output_root_scoped_log(
    monkeypatch, tmp_path: Path
) -> None:
    submitit_renderer = (
        Path(__file__).resolve().parents[1] / "scripts/objv_submitit_batch_render.py"
    )
    tree = ast.parse(submitit_renderer.read_text(encoding="utf-8"))
    render_task = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_render_task"
    )

    output_root = tmp_path / "explicit-output"
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    commands: list[list[str]] = []

    class FakeProcess:
        def wait(self) -> None:
            return None

    def popen(command, **kwargs):
        assert kwargs == {"shell": False, "stdout": fake_subprocess.DEVNULL}
        commands.append(command)
        return FakeProcess()

    fake_subprocess = types.SimpleNamespace(DEVNULL=object(), Popen=popen)
    namespace = {
        "CONFIG": {
            "blender_script_path": "objv_render_vary_intrinsics.py",
            "output_dir": str(output_root),
            "blender_args": {
                "num_views": "8",
                "min_fov": "20.0",
                "max_fov": "80.0",
                "target_coverage": "0.6",
                "seed": "1",
                "resolution_x": "256",
                "resolution_y": "256",
                "engine": "BLENDER_EEVEE",
                "save_mask": False,
                "fix_radial": False,
                "render_depth_only": False,
                "render_depth": True,
            },
        },
        "Path": Path,
        "subprocess": fake_subprocess,
    }
    exec(
        compile(
            ast.Module(body=[render_task], type_ignores=[]),
            str(submitit_renderer),
            "exec",
        ),
        namespace,
    )

    namespace["_render_task"](["/assets/objects/chair.glb"])

    command = commands.pop()
    dump_log = Path(command[command.index("--dump_log") + 1])

    assert dump_log.is_relative_to(output_root)
    assert dump_log.parent.is_dir()
    assert not tuple(unrelated_cwd.iterdir())

    namespace["CONFIG"]["output_dir"] = ""

    with pytest.raises(ValueError, match=r"CONFIG\['output_dir'\] must be set"):
        namespace["_render_task"](["/assets/objects/chair.glb"])

    assert not tuple(unrelated_cwd.iterdir())


def test_re10k_selection_does_not_require_co3d_or_objaverse_environment(monkeypatch) -> None:
    for key in ("OBJV_DIR", "CO3D_DIR", "CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RE10K_TRAIN_DIR", "/synthetic/re10k/train")
    monkeypatch.setenv("RE10K_TEST_DIR", "/synthetic/re10k/test")
    sys.modules.pop("nvs.trainval", None)

    module = importlib.import_module("nvs.trainval")

    assert module.dataset_paths_for("re10k").train == "/synthetic/re10k/train"


def test_co3d_selection_reports_only_the_missing_co3d_variables(monkeypatch) -> None:
    from nvs.runtime_paths import DatasetEnvironmentError, dataset_paths_for

    for key in ("CO3D_DIR", "CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(DatasetEnvironmentError) as raised:
        dataset_paths_for("co3d")

    assert raised.value.missing == ("CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR", "CO3D_DIR")


def test_selected_dataset_reads_only_its_declared_environment() -> None:
    from nvs.runtime_paths import dataset_paths_for

    objaverse = dataset_paths_for("objaverse", {"OBJV_DIR": "/synthetic/objaverse"})
    co3d = dataset_paths_for(
        "co3d",
        {
            "CO3D_DIR": "/synthetic/co3d/images",
            "CO3D_ANNOTATION_DIR": "/synthetic/co3d/annotations",
            "CO3D_DEPTH_DIR": "/synthetic/co3d/depth",
        },
    )

    assert objaverse.root == "/synthetic/objaverse"
    assert co3d.root == "/synthetic/co3d/images"
    assert co3d.annotation == "/synthetic/co3d/annotations"
    assert co3d.depth == "/synthetic/co3d/depth"


def test_selected_dataset_rejects_relative_paths() -> None:
    from nvs.runtime_paths import DatasetEnvironmentError, dataset_paths_for

    with pytest.raises(
        DatasetEnvironmentError, match=r"RE10K_TRAIN_DIR.*relative/train"
    ):
        dataset_paths_for(
            "re10k",
            {
                "RE10K_TRAIN_DIR": "relative/train",
                "RE10K_TEST_DIR": "/synthetic/re10k/test",
            },
        )


def test_default_co3d_evaluation_index_is_module_absolute_when_cwd_changes(
    monkeypatch, tmp_path: Path
) -> None:
    import nvs.trainval as trainval

    monkeypatch.chdir(tmp_path)

    config = trainval.LVSMLauncherConfig()

    expected = (
        Path(trainval.__file__).resolve().parent.parent
        / "assets/co3d_test_context2_seen.json"
    )
    assert Path(config.co3d_test_seen_index_file) == expected
    assert Path(config.co3d_test_seen_index_file).is_absolute()


def test_required_wandb_rejects_non_online_launcher_configuration(tmp_path) -> None:
    from pos_enc.utils.runner import Launcher, LauncherConfig

    config = LauncherConfig(
        output_dir=str(tmp_path / "output"),
        wandb_enabled=True,
        wandb_mode="offline",
        wandb_required=True,
    )

    with pytest.raises(ValueError, match="enabled in online mode"):
        Launcher(config)


def test_launcher_config_exposes_stable_wandb_resume_identity() -> None:
    from pos_enc.utils.runner import LauncherConfig

    config = LauncherConfig(wandb_id="stable-id", wandb_resume="must")
    assert config.wandb_id == "stable-id"
    assert config.wandb_resume == "must"


def test_target_view_chunking_preserves_order_and_bounds_model_calls() -> None:
    import torch

    from nvs.trainval import LVSMLauncher
    from pos_enc.utils.functional import Camera

    launcher = object.__new__(LVSMLauncher)
    launcher.config = types.SimpleNamespace(test_target_view_chunk_size=1)
    calls = []

    class Model:
        def __call__(self, ref_imgs, ref_cams, tar_cams, **kwargs):
            calls.append(tar_cams.camtoworld.shape[1])
            values = tar_cams.K[:, :, 0, 0]
            return values[:, :, None, None, None]

    ref_cams = Camera(
        K=torch.eye(3)[None, None],
        camtoworld=torch.eye(4)[None, None],
        width=1,
        height=1,
    )
    target_K = torch.eye(3)[None, None].repeat(1, 3, 1, 1)
    target_K[0, :, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    tar_cams = Camera(
        K=target_K,
        camtoworld=torch.eye(4)[None, None].repeat(1, 3, 1, 1),
        width=1,
        height=1,
    )

    outputs = launcher._forward_test_target_views(
        Model(), torch.zeros(1, 1, 1, 1, 3), ref_cams, tar_cams
    )

    assert calls == [1, 1, 1]
    assert outputs.flatten().tolist() == [1.0, 2.0, 3.0]


def test_nvs_evaluation_runs_in_inference_mode() -> None:
    import torch

    from nvs.trainval import LVSMLauncher

    launcher = object.__new__(LVSMLauncher)
    launcher.config = types.SimpleNamespace(
        pose_noise_enabled=False,
        render_video=True,
        render_view=False,
    )
    launcher.world_rank = 0
    observed = []

    class Model:
        def eval(self) -> None:
            return None

    class EmptyLoader:
        def __iter__(self):
            observed.append(torch.is_inference_mode_enabled())
            return iter(())

    launcher.test_iteration(
        0,
        {
            "model": Model(),
            "dataloaders": {"4": (4, EmptyLoader())},
        },
    )

    assert observed == [True]
