# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional SwanLab backend for Lightning Fabric, without a W&B dependency."""

from argparse import Namespace
from pathlib import Path

from lightning.fabric.loggers import Logger
from lightning.fabric.utilities.rank_zero import rank_zero_only


class SwanLabLogger(Logger):
    _protomotions_backend = "swanlab"

    def __init__(
        self, project, experiment_name, save_dir, workspace=None, mode=None, id=None
    ):
        # Fail early only when this optional logger was explicitly requested.
        try:
            import swanlab  # noqa: F401
        except ImportError as exc:
            raise ModuleNotFoundError(
                'SwanLab logging requires: pip install -e ".[swanlab]"'
            ) from exc
        self._save_dir = str(save_dir)
        self._project = project
        self._id = id
        self._experiment = None
        self._init_kwargs = {
            "project": project,
            "name": experiment_name,
            "log_dir": str(Path(save_dir) / "swanlog"),
            "id": id,
            "resume": "allow" if id else None,
        }
        if workspace is not None:
            self._init_kwargs["workspace"] = workspace
        if mode is not None:
            self._init_kwargs["mode"] = mode

    @property
    def name(self):
        return self._project

    @property
    def version(self):
        return self._id

    @property
    def root_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return str(Path(self._save_dir) / "swanlog")

    @property
    def experiment(self):
        # Fabric constructs loggers on every rank, but only rank zero may own
        # an SDK run, including when properties are accessed directly.
        if rank_zero_only.rank != 0:
            raise RuntimeError("Only global rank zero may initialize SwanLab")
        if self._experiment is None:
            import swanlab

            self._experiment = swanlab.init(**self._init_kwargs)
            self._id = self._experiment.id
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        if isinstance(params, Namespace):
            params = vars(params)
        self.experiment.config.update(dict(params))

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        # Fabric normally converts scalar tensors; keep direct calls safe too.
        values = {
            key: value.item() if hasattr(value, "item") else value
            for key, value in metrics.items()
        }
        self.experiment.log(values, step=step)

    @rank_zero_only
    def log_video(self, key, path, caption, metrics, step):
        import swanlab

        # GIF encoding happens in the existing background renderer, never in
        # this training-process logging call.
        self.experiment.log(
            {key: swanlab.Video(str(path), caption=caption), **metrics}, step=step
        )

    @rank_zero_only
    def finalize(self, status):
        if self._experiment is not None:
            self._experiment.finish(
                state="success" if status == "success" else "crashed"
            )
            self._experiment = None
