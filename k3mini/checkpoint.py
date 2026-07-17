from __future__ import annotations

import os
import random
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from .config import DataConfig, ModelConfig, TrainConfig


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class CheckpointManager:
    def __init__(self, output_dir: str | Path, rank: int, world_size: int) -> None:
        self.root = Path(output_dir) / "checkpoints"
        self.rank = rank
        self.world_size = world_size

    def save(
        self,
        *,
        consumed_tokens: int,
        update: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None,
        data_stream: Any,
        model_config: ModelConfig,
        data_config: DataConfig,
        train_config: TrainConfig,
    ) -> Path:
        checkpoint_dir = self.root / f"tokens_{consumed_tokens:012d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            _atomic_torch_save(
                {
                    "format_version": 1,
                    "consumed_tokens": consumed_tokens,
                    "update": update,
                    "world_size": self.world_size,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "model_config": asdict(model_config),
                    "data_config": asdict(data_config),
                    "train_config": asdict(train_config),
                },
                checkpoint_dir / "state.pt",
            )
        _atomic_torch_save(
            {
                "format_version": 1,
                "rank": self.rank,
                "world_size": self.world_size,
                "rng": capture_rng_state(),
                "data_stream": data_stream.state_dict(),
            },
            checkpoint_dir / f"rank_{self.rank:04d}.pt",
        )
        _barrier()
        if self.rank == 0:
            (checkpoint_dir / "COMPLETE").write_text("ok\n")
            latest_tmp = self.root / ".LATEST.tmp"
            latest_tmp.write_text(checkpoint_dir.name + "\n")
            os.replace(latest_tmp, self.root / "LATEST")
        _barrier()
        return checkpoint_dir

    def latest(self) -> Path | None:
        latest = self.root / "LATEST"
        if not latest.exists():
            return None
        checkpoint_dir = self.root / latest.read_text().strip()
        if not (checkpoint_dir / "COMPLETE").exists():
            raise RuntimeError(f"incomplete checkpoint referenced by {latest}")
        return checkpoint_dir

    def load(
        self,
        checkpoint: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scaler: torch.amp.GradScaler | None,
        data_stream: Any | None,
    ) -> dict[str, int]:
        path = Path(checkpoint)
        if path.name == "latest":
            resolved = self.latest()
            if resolved is None:
                raise FileNotFoundError(f"no checkpoints below {self.root}")
            path = resolved
        if path.is_file():
            path = path.parent
        shared = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
        if int(shared["world_size"]) != self.world_size:
            raise ValueError(
                f"exact resume requires world_size={shared['world_size']}, got {self.world_size}"
            )
        model.load_state_dict(shared["model"])
        if optimizer is not None:
            optimizer.load_state_dict(shared["optimizer"])
        if scaler is not None and shared["scaler"] is not None:
            scaler.load_state_dict(shared["scaler"])
        if data_stream is not None:
            rank_state = torch.load(
                path / f"rank_{self.rank:04d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            data_stream.load_state_dict(rank_state["data_stream"])
            restore_rng_state(rank_state["rng"])
        return {
            "consumed_tokens": int(shared["consumed_tokens"]),
            "update": int(shared["update"]),
        }
