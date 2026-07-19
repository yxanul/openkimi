from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import DataConfig, ModelConfig, TrainConfig


def _finite_float(value: Any) -> float | int | str | bool:
    if isinstance(value, torch.Tensor):
        value = value.detach().float().item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class WandbTracker:
    """Rank-zero W&B logging with token-based steps and resumable run IDs."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str | Path,
        model_config: ModelConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
        startup: Mapping[str, Any],
    ) -> None:
        self.run: Any | None = None
        self._wandb: Any | None = None
        self.histogram_every = train_config.wandb_router_histogram_every_logs
        self.log_count = 0
        if not enabled or not train_config.wandb_enabled:
            return

        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("W&B tracking requires the locked `wandb` dependency") from error

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        run_id_path = root / ".wandb-run-id"
        run_id = train_config.wandb_run_id
        if run_id is None and run_id_path.exists():
            run_id = run_id_path.read_text().strip()
        if not run_id:
            run_id = wandb.util.generate_id()
            run_id_path.write_text(run_id + "\n")

        self._wandb = wandb
        self.run = wandb.init(
            project=train_config.wandb_project,
            entity=train_config.wandb_entity,
            name=train_config.wandb_run_name,
            id=run_id,
            resume="allow",
            mode=train_config.wandb_mode,
            tags=list(train_config.wandb_tags),
            dir=str(root),
            config={
                "model": asdict(model_config),
                "data": asdict(data_config),
                "train": asdict(train_config),
                "startup": dict(startup),
            },
        )
        self.run.define_metric("tokens")
        self.run.define_metric("*", step_metric="tokens")

    def log_train(
        self,
        metrics: Mapping[str, Any],
        router: Mapping[str, Any],
    ) -> None:
        if self.run is None:
            return
        payload = {
            f"train/{key}": _finite_float(value)
            for key, value in metrics.items()
            if key not in {"tokens", "update"}
        }
        payload["tokens"] = int(metrics["tokens"])
        payload["update"] = int(metrics["update"])
        for key, value in router.items():
            if key == "expert_loads":
                continue
            payload[f"router/{key}"] = _finite_float(value)

        self.log_count += 1
        loads = router.get("expert_loads", [])
        if loads and self.log_count % self.histogram_every == 0:
            flattened = [load for layer in loads for load in layer]
            payload["router/load_histogram"] = self._wandb.Histogram(flattened)
            for layer_idx, layer_loads in enumerate(loads):
                payload[f"router/layer_{layer_idx:02d}_load_histogram"] = (
                    self._wandb.Histogram(layer_loads)
                )
        self.run.log(payload)

    def log_validation(self, tokens: int, metrics: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        self.run.log(
            {
                "tokens": tokens,
                **{
                    f"validation/{key}": _finite_float(value)
                    for key, value in metrics.items()
                },
            }
        )

    def log_evaluation(self, tokens: int, metrics: Mapping[str, Any]) -> None:
        if self.run is None:
            return
        self.run.log(
            {
                "tokens": tokens,
                **{
                    f"eval/{key}": _finite_float(value)
                    for key, value in metrics.items()
                },
            }
        )

    def finish(self, *, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)
