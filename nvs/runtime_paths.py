"""Resolve the data paths required by one selected NVS dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class DatasetPaths:
    """The environment-backed paths for a single dataset selection."""

    dataset: str
    train: Optional[str] = None
    test: Optional[str] = None
    root: Optional[str] = None
    annotation: Optional[str] = None
    depth: Optional[str] = None


class DatasetEnvironmentError(RuntimeError):
    """Raised when the selected dataset is missing required environment paths."""

    dataset: str
    missing: tuple[str, ...]

    def __init__(self, dataset: str, missing: tuple[str, ...]) -> None:
        self.dataset = dataset
        self.missing = tuple(sorted(missing))
        missing_names = ", ".join(self.missing)
        super().__init__(f"Dataset {dataset!r} requires environment variables: {missing_names}")


_REQUIRED_VARIABLES = {
    "re10k": ("RE10K_TRAIN_DIR", "RE10K_TEST_DIR"),
    "objaverse": ("OBJV_DIR",),
    "co3d": ("CO3D_DIR", "CO3D_ANNOTATION_DIR", "CO3D_DEPTH_DIR"),
}


def dataset_paths_for(
    dataset: str, environ: Optional[Mapping[str, str]] = None
) -> DatasetPaths:
    """Return paths for ``dataset`` after validating only its required variables."""

    try:
        required = _REQUIRED_VARIABLES[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset!r}") from error

    environment = os.environ if environ is None else environ
    values = {name: environment.get(name) for name in required}
    missing = tuple(name for name in required if not values[name])
    if missing:
        raise DatasetEnvironmentError(dataset, missing)

    if dataset == "re10k":
        return DatasetPaths(
            dataset=dataset,
            train=values["RE10K_TRAIN_DIR"],
            test=values["RE10K_TEST_DIR"],
        )
    if dataset == "objaverse":
        return DatasetPaths(dataset=dataset, root=values["OBJV_DIR"])
    return DatasetPaths(
        dataset=dataset,
        root=values["CO3D_DIR"],
        annotation=values["CO3D_ANNOTATION_DIR"],
        depth=values["CO3D_DEPTH_DIR"],
    )
