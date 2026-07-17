from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar


class KernelBackend(StrEnum):
    AUTO = "auto"
    REFERENCE = "reference"
    H100 = "h100"


class LossBackend(StrEnum):
    AUTO = "auto"
    TORCH = "torch"
    FLA = "fla"
    LIGER = "liger"


class LinearPrecision(StrEnum):
    BF16 = "bf16"
    FP8_CURRENT = "fp8_current"


class RouterType(StrEnum):
    SOFTMAX = "softmax"
    SIGMOID_NOAUX = "sigmoid_noaux"


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int = 128_001
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 6
    kda_head_dim: int = 128
    kda_conv_kernel: int = 4
    global_attn_every: int = 4
    mla_qk_head_dim: int = 128
    mla_v_head_dim: int = 128
    mla_kv_lora_rank: int = 192
    attnres_block_size: int = 4
    rms_norm_eps: float = 1e-6
    latent_dim: int = 192
    n_routed_experts: int = 64
    top_k: int = 4
    expert_ffn_dim: int = 512
    n_shared_experts: int = 1
    shared_ffn_dim: int = 512
    dense_ffn_dim: int = 512
    router_type: RouterType = RouterType.SOFTMAX
    router_scale: float = 1.0
    router_aux_loss_coefficient: float = 0.01
    router_z_loss_coefficient: float = 0.001
    router_bias_update_rate: float = 1e-3
    router_num_groups: int = 1
    router_topk_groups: int = 1
    dropout: float = 0.0
    tie_embeddings: bool = True
    kernel_backend: KernelBackend = KernelBackend.AUTO
    loss_backend: LossBackend = LossBackend.AUTO
    linear_precision: LinearPrecision = LinearPrecision.BF16
    activation_checkpointing: bool = True

    def __post_init__(self) -> None:
        self.router_type = RouterType(self.router_type)
        self.kernel_backend = KernelBackend(self.kernel_backend)
        self.loss_backend = LossBackend(self.loss_backend)
        self.linear_precision = LinearPrecision(self.linear_precision)

    @property
    def is_primary_shape(self) -> bool:
        return self.d_model == 768 and self.n_heads == 6 and self.n_layers == 16

    @property
    def physical_vocab_size(self) -> int:
        if self.linear_precision is LinearPrecision.FP8_CURRENT:
            return ((self.vocab_size + 15) // 16) * 16
        return self.vocab_size

    def validate(self) -> None:
        if self.n_heads * self.kda_head_dim != self.d_model:
            raise ValueError("n_heads * kda_head_dim must equal d_model")
        if self.global_attn_every < 0:
            raise ValueError("global_attn_every must be non-negative")
        if self.kda_conv_kernel < 1:
            raise ValueError("kda_conv_kernel must be positive")
        if not 1 <= self.top_k <= self.n_routed_experts:
            raise ValueError("top_k must be in [1, n_routed_experts]")
        if not 0 < self.latent_dim <= self.d_model:
            raise ValueError("latent_dim must be in (0, d_model]")
        if self.attnres_block_size < 1:
            raise ValueError("attnres_block_size must be positive")
        if self.n_routed_experts % self.router_num_groups:
            raise ValueError("n_routed_experts must be divisible by router_num_groups")
        if not 1 <= self.router_topk_groups <= self.router_num_groups:
            raise ValueError("router_topk_groups must be in [1, router_num_groups]")
        if self.kernel_backend is KernelBackend.H100 and self.kda_head_dim != 128:
            raise ValueError("the optimized H100 KDA profile requires kda_head_dim=128")
        if self.linear_precision is LinearPrecision.FP8_CURRENT:
            if self.kernel_backend is KernelBackend.REFERENCE:
                raise ValueError("linear_precision=fp8_current requires the H100 kernel backend")
            if not self.tie_embeddings:
                raise ValueError("linear_precision=fp8_current currently requires tied embeddings")
            fp8_dimensions = {
                "d_model": self.d_model,
                "latent_dim": self.latent_dim,
                "dense_ffn_dim": self.dense_ffn_dim,
                "shared_ffn_dim": self.shared_ffn_dim,
            }
            misaligned = [name for name, size in fp8_dimensions.items() if size % 16]
            if misaligned:
                raise ValueError(
                    "linear_precision=fp8_current requires dimensions divisible by 16: "
                    + ", ".join(misaligned)
                )
        if self.dropout != 0:
            raise ValueError("the public-faithful profile uses zero dropout")


@dataclass(slots=True)
class DataConfig:
    dataset_name: str = "OptimalScale/ClimbMix"
    dataset_config: str | None = None
    dataset_split: str = "train"
    dataset_revision: str = "main"
    tokenizer_name: str = "alisawuffles/superbpe-tokenizer-128k"
    tokenizer_revision: str = "main"
    sequence_length: int = 4096
    eod_token_id: int = 128_000
    shuffle_buffer_size: int = 10_000
    tokenizer_batch_size: int = 64
    validation_fraction: float = 0.001
    validation_tokens: int = 1_000_000
    validation_cache: str = "data/climbmix-validation-1m.pt"
    seed: int = 1234
    num_workers: int = 0

    def validate(self) -> None:
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.num_workers != 0:
            raise ValueError("exact iterable resume currently requires num_workers=0")


@dataclass(slots=True)
class TrainConfig:
    output_dir: str = "runs/k3-mini"
    target_tokens: int = 1_000_000_000
    global_batch_tokens: int = 262_144
    microbatch_sequences: int = 1
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip: float = 1.0
    warmup_updates: int = 100
    validate_every_tokens: int = 25_000_000
    checkpoint_every_tokens: int = 50_000_000
    log_every_updates: int = 5
    precision: str = "bf16"
    seed: int = 1234
    compile_model: bool = False

    def __post_init__(self) -> None:
        self.betas = tuple(self.betas)  # type: ignore[assignment]

    def validate(self, data: DataConfig, world_size: int = 1) -> None:
        tokens_per_microbatch = data.sequence_length * self.microbatch_sequences * world_size
        if self.global_batch_tokens % tokens_per_microbatch:
            raise ValueError(
                "global_batch_tokens must be divisible by sequence_length * microbatch_sequences * world_size"
            )
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("precision must be bf16 or fp32")
        if self.target_tokens < self.global_batch_tokens:
            raise ValueError("target_tokens must be at least one global batch")

    def gradient_accumulation(self, data: DataConfig, world_size: int) -> int:
        self.validate(data, world_size)
        return self.global_batch_tokens // (data.sequence_length * self.microbatch_sequences * world_size)


T = TypeVar("T")


def _from_mapping(cls: type[T], values: dict[str, Any] | None) -> T:
    values = values or {}
    allowed = {field.name for field in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> tuple[ModelConfig, DataConfig, TrainConfig]:
    payload = json.loads(Path(path).read_text())
    model = _from_mapping(ModelConfig, payload.get("model"))
    data = _from_mapping(DataConfig, payload.get("data"))
    train = _from_mapping(TrainConfig, payload.get("train"))
    model.validate()
    data.validate()
    return model, data, train


def save_config(
    path: str | Path,
    model: ModelConfig,
    data: DataConfig,
    train: TrainConfig,
) -> None:
    payload = {"model": asdict(model), "data": asdict(data), "train": asdict(train)}
    Path(path).write_text(json.dumps(payload, indent=2, default=str) + "\n")
