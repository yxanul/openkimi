from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .config import DataConfig, ModelConfig, TrainConfig, save_config
from .data import (
    PackedClimbMixDataset,
    SyntheticTokenDataset,
    load_validation_cache,
)
from .evaluation import run_lm_evaluation
from .model import K3MiniForCausalLM, ModelOutput
from .optim import OptimizerLike, build_optimizer
from .tracking import WandbTracker


@dataclass(frozen=True, slots=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = local_rank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    return DistributedContext(distributed, rank, local_rank, world_size, device)


def seed_all(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)


def learning_rate_at_tokens(
    consumed_tokens: int,
    cfg: TrainConfig,
    *,
    peak: float | None = None,
    minimum: float | None = None,
) -> float:
    peak = cfg.learning_rate if peak is None else peak
    minimum = cfg.min_learning_rate if minimum is None else minimum
    warmup_tokens = cfg.warmup_updates * cfg.global_batch_tokens
    if consumed_tokens < warmup_tokens:
        return peak * (consumed_tokens + cfg.global_batch_tokens) / warmup_tokens
    progress = (consumed_tokens - warmup_tokens) / max(1, cfg.target_tokens - warmup_tokens)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak - minimum) * cosine


def _set_learning_rates(
    optimizer: OptimizerLike,
    adam_learning_rate: float,
    muon_learning_rate: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = (
            muon_learning_rate
            if group.get("optimizer_kind") == "muon"
            else adam_learning_rate
        )


def _distributed_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def _autocast_context(device: torch.device, precision: str):
    enabled = precision == "bf16" and device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    samples: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = torch.zeros((), device=device)
    total_tokens = 0
    for sample in samples:
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        labels = sample["labels"].unsqueeze(0).to(device)
        with _autocast_context(device, precision):
            output: ModelOutput = model(input_ids, labels)
        assert output.lm_loss is not None
        tokens = input_ids.numel()
        total_loss += output.lm_loss.float() * tokens
        total_tokens += tokens
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_loss)
        count = torch.tensor(total_tokens, device=device, dtype=torch.long)
        dist.all_reduce(count)
        total_tokens = int(count.item())
    if was_training:
        model.train()
    mean_loss = float((total_loss / max(1, total_tokens)).item())
    return {"loss": mean_loss, "perplexity": math.exp(min(20.0, mean_loss)), "tokens": total_tokens}


def train(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    *,
    resume: str | None = None,
    synthetic: bool = False,
    fixed_batch: bool = False,
) -> dict[str, Any]:
    context = setup_distributed()
    train_cfg.validate(data_cfg, context.world_size)
    seed_all(train_cfg.seed, context.rank)
    if context.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    stream: Any
    if synthetic:
        stream = SyntheticTokenDataset(
            model_cfg.vocab_size, data_cfg.sequence_length, data_cfg.seed + context.rank
        )
    else:
        stream = PackedClimbMixDataset(data_cfg, rank=context.rank, world_size=context.world_size)
    loader = DataLoader(
        stream,
        batch_size=train_cfg.microbatch_sequences,
        num_workers=0,
        pin_memory=context.device.type == "cuda",
    )
    iterator = iter(loader)
    fixed_samples: list[dict[str, torch.Tensor]] | None = None
    if fixed_batch:
        fixed_samples = [next(iterator)]

    raw_model = K3MiniForCausalLM(model_cfg).to(context.device)
    optimizer = build_optimizer(raw_model, train_cfg)
    execution_model: torch.nn.Module = raw_model
    if train_cfg.compile_model:
        execution_model = torch.compile(raw_model)
    model: torch.nn.Module
    if context.distributed:
        model = DistributedDataParallel(
            execution_model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
    else:
        model = execution_model
    scaler: torch.amp.GradScaler | None = None
    manager = CheckpointManager(train_cfg.output_dir, context.rank, context.world_size)
    consumed_tokens = update = 0
    if resume:
        state = manager.load(
            resume,
            model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            data_stream=stream,
        )
        consumed_tokens, update = state["consumed_tokens"], state["update"]
        iterator = iter(loader)

    output_dir = Path(train_cfg.output_dir)
    startup = {
        "device": str(context.device),
        "world_size": context.world_size,
        "gradient_accumulation": train_cfg.gradient_accumulation(
            data_cfg,
            context.world_size,
        ),
        "parameters": raw_model.parameter_counts(),
        "backend": raw_model.backend.as_dict(),
    }
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(output_dir / "config.json", model_cfg, data_cfg, train_cfg)
        print(
            json.dumps(
                {
                    "event": "startup",
                    **startup,
                }
            ),
            flush=True,
        )
    tracker = WandbTracker(
        enabled=context.is_main,
        output_dir=output_dir,
        model_config=model_cfg,
        data_config=data_cfg,
        train_config=train_cfg,
        startup=startup,
    )

    validation_samples: list[dict[str, torch.Tensor]] | None = None
    validation_path = Path(data_cfg.validation_cache)
    if validation_path.exists():
        validation_samples = load_validation_cache(validation_path)
        validation_samples = validation_samples[context.rank :: context.world_size]

    grad_accum = train_cfg.gradient_accumulation(data_cfg, context.world_size)
    next_validation = (
        (consumed_tokens // train_cfg.validate_every_tokens) + 1
    ) * train_cfg.validate_every_tokens
    next_checkpoint = (
        (consumed_tokens // train_cfg.checkpoint_every_tokens) + 1
    ) * train_cfg.checkpoint_every_tokens
    running_loss = 0.0
    running_lm_loss = 0.0
    running_mtp_loss = 0.0
    running_router_aux = 0.0
    running_router_z = 0.0
    running_tokens = 0
    running_updates = 0
    log_start = time.perf_counter()
    training_start = time.monotonic()
    next_eval_time = (
        training_start + train_cfg.eval_every_seconds
        if train_cfg.eval_every_seconds and train_cfg.eval_tasks
        else math.inf
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while consumed_tokens < train_cfg.target_tokens:
        adam_learning_rate = learning_rate_at_tokens(consumed_tokens, train_cfg)
        muon_learning_rate = learning_rate_at_tokens(
            consumed_tokens,
            train_cfg,
            peak=train_cfg.muon_learning_rate,
            minimum=train_cfg.muon_min_learning_rate,
        )
        _set_learning_rates(
            optimizer,
            adam_learning_rate,
            muon_learning_rate,
        )
        step_loss = torch.zeros((), device=context.device)
        step_lm_loss = torch.zeros((), device=context.device)
        step_mtp_loss = torch.zeros((), device=context.device)
        step_router_aux = torch.zeros((), device=context.device)
        step_router_z = torch.zeros((), device=context.device)

        for micro_step in range(grad_accum):
            batch = fixed_samples[0] if fixed_samples is not None else next(iterator)
            input_ids = batch["input_ids"].to(context.device, non_blocking=True)
            labels = batch["labels"].to(context.device, non_blocking=True)
            sync = model.no_sync() if context.distributed and micro_step < grad_accum - 1 else nullcontext()
            with sync, _autocast_context(context.device, train_cfg.precision):
                output: ModelOutput = model(
                    input_ids,
                    labels,
                    is_first_microbatch=micro_step == 0,
                )
                assert output.loss is not None and output.lm_loss is not None
                loss = output.loss / grad_accum
            loss.backward()
            step_loss += output.loss.detach() / grad_accum
            step_lm_loss += output.lm_loss.detach() / grad_accum
            if output.mtp_loss is not None:
                step_mtp_loss += output.mtp_loss.detach() / grad_accum
            step_router_aux += output.router_aux_loss.detach() / grad_accum
            step_router_z += output.router_z_loss.detach() / grad_accum

        gradient_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.gradient_clip)
        gradients_finite = bool(torch.isfinite(gradient_norm).item())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        raw_model.update_router_biases()
        update += 1
        consumed_tokens += train_cfg.global_batch_tokens
        running_tokens += train_cfg.global_batch_tokens
        running_updates += 1
        running_loss += float(_distributed_mean(step_loss))
        running_lm_loss += float(_distributed_mean(step_lm_loss))
        running_mtp_loss += float(_distributed_mean(step_mtp_loss))
        running_router_aux += float(_distributed_mean(step_router_aux))
        running_router_z += float(_distributed_mean(step_router_z))

        should_log = update % train_cfg.log_every_updates == 0 or update == 1
        router_diagnostics = raw_model.router_diagnostics() if should_log else {}
        if context.is_main and should_log:
            # Wall-clock throughput must include all CUDA work enqueued for the
            # update. Synchronize only at the existing logging boundary so the
            # hot path between reports remains asynchronous.
            if context.device.type == "cuda":
                torch.cuda.synchronize(context.device)
            elapsed = max(1e-9, time.perf_counter() - log_start)
            divisor = running_updates
            optimizer_diagnostics = getattr(optimizer, "diagnostics", lambda: {})()
            memory_metrics: dict[str, float] = {}
            if context.device.type == "cuda":
                memory_metrics = {
                    "allocated_gib": torch.cuda.memory_allocated(context.device) / 2**30,
                    "reserved_gib": torch.cuda.memory_reserved(context.device) / 2**30,
                    "max_allocated_gib": torch.cuda.max_memory_allocated(context.device)
                    / 2**30,
                }
                torch.cuda.reset_peak_memory_stats(context.device)
            train_metrics = {
                "update": update,
                "tokens": consumed_tokens,
                "loss": running_loss / divisor,
                "lm_loss": running_lm_loss / divisor,
                "mtp_loss": running_mtp_loss / divisor,
                "router_aux_loss": running_router_aux / divisor,
                "router_z_loss": running_router_z / divisor,
                "adam_learning_rate": adam_learning_rate,
                "muon_learning_rate": muon_learning_rate,
                "gradient_norm": float(gradient_norm),
                "gradients_finite": int(gradients_finite),
                "tokens_per_second": running_tokens / elapsed,
                "elapsed_seconds": time.monotonic() - training_start,
                **optimizer_diagnostics,
                **memory_metrics,
            }
            print(
                json.dumps(
                    {
                        "event": "train",
                        **train_metrics,
                        "router": router_diagnostics,
                    }
                ),
                flush=True,
            )
            tracker.log_train(train_metrics, router_diagnostics)
            running_loss = running_lm_loss = running_mtp_loss = 0.0
            running_router_aux = running_router_z = 0.0
            running_tokens = 0
            running_updates = 0
            log_start = time.perf_counter()

        if consumed_tokens >= next_validation:
            if validation_samples is not None:
                metrics = validate(model, validation_samples, context.device, train_cfg.precision)
                if context.is_main:
                    print(
                        json.dumps({"event": "validation", "tokens": consumed_tokens, **metrics}),
                        flush=True,
                    )
                    tracker.log_validation(consumed_tokens, metrics)
            elif context.is_main:
                print(
                    json.dumps(
                        {
                            "event": "validation_skipped",
                            "reason": f"cache not found: {validation_path}",
                        }
                    ),
                    flush=True,
                )
            next_validation += train_cfg.validate_every_tokens

        if time.monotonic() >= next_eval_time and should_log:
            if context.distributed:
                dist.barrier()
            if context.is_main:
                if context.device.type == "cuda":
                    torch.cuda.synchronize(context.device)
                    torch.cuda.empty_cache()
                eval_start = time.perf_counter()
                evaluation = run_lm_evaluation(
                    raw_model,
                    data_cfg,
                    train_cfg,
                    context.device,
                )
                evaluation["elapsed_seconds"] = time.perf_counter() - eval_start
                print(
                    json.dumps(
                        {
                            "event": "lm_evaluation",
                            "tokens": consumed_tokens,
                            **evaluation,
                        }
                    ),
                    flush=True,
                )
                tracker.log_evaluation(consumed_tokens, evaluation)
                if context.device.type == "cuda":
                    torch.cuda.empty_cache()
            if context.distributed:
                dist.barrier()
            next_eval_time = time.monotonic() + train_cfg.eval_every_seconds
            log_start = time.perf_counter()

        if consumed_tokens >= next_checkpoint:
            manager.save(
                consumed_tokens=consumed_tokens,
                update=update,
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                data_stream=stream,
                model_config=model_cfg,
                data_config=data_cfg,
                train_config=train_cfg,
            )
            next_checkpoint += train_cfg.checkpoint_every_tokens

        if (
            train_cfg.max_duration_seconds is not None
            and time.monotonic() - training_start >= train_cfg.max_duration_seconds
        ):
            break

    checkpoint = manager.save(
        consumed_tokens=consumed_tokens,
        update=update,
        model=raw_model,
        optimizer=optimizer,
        scaler=scaler,
        data_stream=stream,
        model_config=model_cfg,
        data_config=data_cfg,
        train_config=train_cfg,
    )
    if context.distributed:
        dist.destroy_process_group()
    tracker.finish()
    return {
        "consumed_tokens": consumed_tokens,
        "updates": update,
        "checkpoint": str(checkpoint),
        "duration_seconds": time.monotonic() - training_start,
    }
