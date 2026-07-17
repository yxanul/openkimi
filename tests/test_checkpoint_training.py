from __future__ import annotations

import torch

from k3mini.checkpoint import CheckpointManager
from k3mini.data import SyntheticTokenDataset
from k3mini.model import K3MiniForCausalLM
from k3mini.training import build_optimizer, train


def test_checkpoint_restores_stream_and_model(tiny_model_config, tiny_data_config, tiny_train_config) -> None:
    model = K3MiniForCausalLM(tiny_model_config)
    optimizer = build_optimizer(model, tiny_train_config)
    stream = SyntheticTokenDataset(tiny_model_config.vocab_size, tiny_data_config.sequence_length)
    iterator = iter(stream)
    next(iterator)
    manager = CheckpointManager(tiny_train_config.output_dir, rank=0, world_size=1)
    checkpoint = manager.save(
        consumed_tokens=64,
        update=1,
        model=model,
        optimizer=optimizer,
        scaler=None,
        data_stream=stream,
        model_config=tiny_model_config,
        data_config=tiny_data_config,
        train_config=tiny_train_config,
    )
    expected = next(iterator)

    restored_model = K3MiniForCausalLM(tiny_model_config)
    restored_optimizer = build_optimizer(restored_model, tiny_train_config)
    restored_stream = SyntheticTokenDataset(tiny_model_config.vocab_size, tiny_data_config.sequence_length)
    state = manager.load(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=None,
        data_stream=restored_stream,
    )
    actual = next(iter(restored_stream))
    assert state == {"consumed_tokens": 64, "update": 1}
    torch.testing.assert_close(expected["input_ids"], actual["input_ids"])
    for expected_parameter, actual_parameter in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected_parameter, actual_parameter)


def test_one_update_synthetic_training(tiny_model_config, tiny_data_config, tiny_train_config) -> None:
    result = train(
        tiny_model_config,
        tiny_data_config,
        tiny_train_config,
        synthetic=True,
    )
    assert result["consumed_tokens"] == 64
    assert result["updates"] == 1
