from __future__ import annotations

import pytest

from k3mini.config import DataConfig, KernelBackend, ModelConfig, TrainConfig


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=67,
        d_model=32,
        n_layers=4,
        n_heads=4,
        kda_head_dim=8,
        kda_conv_kernel=3,
        global_attn_every=4,
        mla_qk_head_dim=8,
        mla_v_head_dim=8,
        mla_kv_lora_rank=8,
        attnres_block_size=2,
        latent_dim=8,
        n_routed_experts=4,
        top_k=2,
        expert_ffn_dim=16,
        shared_ffn_dim=24,
        dense_ffn_dim=24,
        kernel_backend=KernelBackend.REFERENCE,
        activation_checkpointing=False,
        mtp_depth=0,
    )


@pytest.fixture
def tiny_data_config() -> DataConfig:
    return DataConfig(
        sequence_length=8,
        eod_token_id=66,
        shuffle_buffer_size=8,
        tokenizer_batch_size=4,
        validation_tokens=64,
    )


@pytest.fixture
def tiny_train_config(tmp_path) -> TrainConfig:
    return TrainConfig(
        output_dir=str(tmp_path / "run"),
        target_tokens=64,
        global_batch_tokens=64,
        microbatch_sequences=1,
        warmup_updates=2,
        validate_every_tokens=64,
        checkpoint_every_tokens=64,
        log_every_updates=1,
        precision="fp32",
    )
