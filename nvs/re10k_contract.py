"""RayRoPE-owned RE10K adapter for the shared dataset contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rope_contract.dataset.migration import (
    DatasetMigrationDispatcher,
    DatasetRuntimeBinding,
    FrozenSamplePlan,
    source_snapshot_digest,
)

from .re10k_dataset import (
    RE10K_EvalDataset,
    RE10K_TrainDataset,
    _normalize_poses_identity_unit_distance,
    _resolve_re10k_eval_index_file,
    load_and_maybe_update_meta_info,
    load_frames_from_meta_info,
)


CONSUMER_ADAPTER_ID = "rayrope-multiview-nvs-v1"


def _consumer_source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = ("nvs/re10k_contract.py", "nvs/trainval.py")
    return source_snapshot_digest({path: root / path for path in paths})


def _binding() -> DatasetRuntimeBinding:
    return DatasetRuntimeBinding.from_environment(
        expected_consumer_adapter_id=CONSUMER_ADAPTER_ID
    )


def _scene_metadata(scene) -> Dict[str, Any]:
    first = scene.frames[0].camera
    width, height = first.image_size
    fx, _, cx, _, fy, cy, _, _, _ = first.intrinsics
    frames = []
    for frame in scene.frames:
        if frame.image is None:
            raise ValueError(f"contract scene frame has no image: {frame.frame_id}")
        if frame.camera.image_size != (width, height):
            raise ValueError("RayRoPE requires one image size per RE10K scene")
        matrix = frame.camera.camera_to_world
        frames.append({
            "file_path": frame.image.uri,
            "transform_matrix": [list(matrix[row * 4 : row * 4 + 4]) for row in range(4)],
        })
    return {
        "w": width,
        "h": height,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": frames,
    }


def _parity_projection(loaded: Dict[str, Any]) -> Dict[str, Any]:
    return {key: loaded[key] for key in ("image", "K", "camtoworld")}


def _sample_id(scene_id: str, frame_ids) -> str:
    payload = f"{scene_id}:{','.join(str(int(value)) for value in frame_ids)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class _ContractMixin:
    binding: DatasetRuntimeBinding
    provider: Any
    split: str

    def _contract_metadata(self, scene_id: str) -> Dict[str, Any]:
        scene = self.provider.load_scene(
            scene_id, self.binding.profile_id, split=self.split
        )
        return _scene_metadata(scene)

    def _load_selected(
        self,
        scene_id: str,
        frame_ids,
        *,
        legacy_scene_dir: Path,
        patch_size: int,
        zoom_factor: float,
        random_zoom: bool,
    ) -> Dict[str, Any]:
        random_state = np.random.get_state()
        contract = load_frames_from_meta_info(
            str(self.provider.root),
            self._contract_metadata(scene_id),
            frame_ids,
            patch_size=patch_size,
            zoom_factor=zoom_factor,
            random_zoom=random_zoom,
        )
        if self.binding.mode == "contract":
            return contract
        valid, legacy_metadata = load_and_maybe_update_meta_info(
            str(legacy_scene_dir / "transforms.json")
        )
        if not valid:
            raise ValueError(f"invalid legacy scene for compare mode: {legacy_scene_dir}")
        np.random.set_state(random_state)
        legacy = load_frames_from_meta_info(
            str(legacy_scene_dir),
            legacy_metadata,
            frame_ids,
            patch_size=patch_size,
            zoom_factor=zoom_factor,
            random_zoom=random_zoom,
        )
        artifact_root = os.environ.get("WORKSPACE_ARTIFACT_DIR")
        if not artifact_root or not os.path.isabs(artifact_root):
            raise ValueError("compare mode requires an absolute WORKSPACE_ARTIFACT_DIR")
        sample_id = _sample_id(scene_id, frame_ids)
        dispatcher = DatasetMigrationDispatcher(
            mode="compare",
            plan=FrozenSamplePlan((sample_id,)),
            legacy_loader=lambda _: _parity_projection(legacy),
            contract_loader=lambda _: _parity_projection(contract),
            dataset_id=self.binding.dataset_id,
            provider_id=self.binding.provider_id,
            profile_id=self.binding.profile_id,
            consumer_adapter_id=self.binding.consumer_adapter_id,
            receipt_dir=Path(artifact_root) / "dataset-contract-receipts" / CONSUMER_ADAPTER_ID,
            absolute_tolerance=1e-5,
            relative_tolerance=1e-6,
            consumer_source_digest=_consumer_source_digest(),
        )
        dispatcher.load(sample_id)
        return legacy


class ContractRE10KTrainDataset(_ContractMixin, RE10K_TrainDataset):
    def __init__(self, data_dirs: List[str], **kwargs) -> None:
        self.binding = _binding()
        if self.binding.mode not in {"contract", "compare"}:
            raise ValueError("contract adapter requires contract or compare mode")
        self.provider = self.binding.create_provider()
        self.split = "train"
        scene_ids = [Path(value).name for value in data_dirs]
        super().__init__(scene_ids, **kwargs)

    def __getitem__(self, _: Any) -> Dict[str, Any]:
        scene_id = str(np.random.choice(self.data_dirs), encoding="utf-8")
        metadata = self._contract_metadata(scene_id)
        frame_ids = self._select_views(len(metadata["frames"]))
        if frame_ids is None:
            return self.__getitem__(None)
        legacy_root = Path(os.environ.get("RE10K_TRAIN_DIR", str(self.provider.root / "train")))
        loaded = self._load_selected(
            scene_id,
            frame_ids,
            legacy_scene_dir=legacy_root / scene_id,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=self.random_zoom,
        )
        camtoworld = _normalize_poses_identity_unit_distance(
            torch.from_numpy(loaded["camtoworld"]).float(),
            ref0_idx=0,
            ref1_idx=self.input_views - 1,
        )
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(loaded["K"]).float(),
            "image": torch.from_numpy(loaded["image"]).float(),
            "image_path": loaded["image_path"],
        }


class ContractRE10KEvalDataset(_ContractMixin, RE10K_EvalDataset):
    def __init__(
        self,
        folder: str,
        patch_size: int = 256,
        zoom_factor: float = 1.0,
        random_zoom: bool = False,
        verbose: bool = False,
        first_n: Optional[int] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        input_views: int = 2,
        supervise_views: int = 3,
        render_video: bool = False,
        test_index_fp: Optional[str] = None,
    ) -> None:
        self.binding = _binding()
        if self.binding.mode not in {"contract", "compare"}:
            raise ValueError("contract adapter requires contract or compare mode")
        self.provider = self.binding.create_provider()
        self.split = "test"
        self.patch_size = patch_size
        self.zoom_factor = zoom_factor
        self.random_zoom = random_zoom
        self.input_views = input_views
        self.supervise_views = supervise_views
        self.render_video = render_video
        if test_index_fp is None:
            if render_video:
                test_index_fp = "evaluation_index_re10k_video.json"
            elif (input_views, supervise_views) == (2, 3):
                test_index_fp = "evaluation_index_re10k.json"
            elif (input_views, supervise_views) == (4, 3):
                test_index_fp = "evaluation_index_re10k_4ctx.json"
            else:
                raise ValueError("unsupported RE10K eval view configuration")
        index_path = Path(_resolve_re10k_eval_index_file(test_index_fp))
        index_info = json.loads(index_path.read_text(encoding="utf-8"))
        available = set(self.provider.list_scene_ids(split="test"))
        scenes = sorted(
            scene for scene, selection in index_info.items()
            if selection is not None and scene in available
        )
        if verbose:
            print(f"[ContractRE10KEvalDataset] Using {len(scenes)} scenes.")
        if first_n is not None:
            scenes = scenes[:first_n]
        if rank is not None and world_size is not None:
            scenes = scenes[rank::world_size]
        self.scene_ids = np.array(scenes).astype(np.bytes_)
        self.data_dirs = self.scene_ids
        self.contexts = np.array([index_info[scene]["context"] for scene in scenes])
        self.targets = (
            [index_info[scene]["target"] for scene in scenes]
            if render_video
            else np.array([index_info[scene]["target"] for scene in scenes])
        )
        self.legacy_root = Path(folder)

    def __getitem__(self, scene_index: int) -> Dict[str, Any]:
        scene_id = str(self.scene_ids[scene_index], encoding="utf-8")
        context = self.contexts[scene_index][: self.input_views]
        target = self.targets[scene_index] if self.render_video else self.targets[scene_index][: self.supervise_views]
        frame_ids = np.concatenate([context, target])
        loaded = self._load_selected(
            scene_id,
            frame_ids,
            legacy_scene_dir=self.legacy_root / scene_id,
            patch_size=self.patch_size,
            zoom_factor=self.zoom_factor,
            random_zoom=self.random_zoom,
        )
        camtoworld = _normalize_poses_identity_unit_distance(
            torch.from_numpy(loaded["camtoworld"]).float(),
            ref0_idx=0,
            ref1_idx=self.input_views - 1,
        )
        return {
            "camtoworld": camtoworld,
            "K": torch.from_numpy(loaded["K"]).float(),
            "image": torch.from_numpy(loaded["image"]).float(),
            "image_path": loaded["image_path"],
            "scene": scene_index,
        }


__all__ = ["ContractRE10KTrainDataset", "ContractRE10KEvalDataset"]
