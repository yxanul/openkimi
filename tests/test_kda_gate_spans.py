from __future__ import annotations

import math

import torch

from scripts.profile_kda_gate_spans import gate_tile_spans


def test_gate_tile_spans_match_manual_log2_cumsum() -> None:
    raw_decay = torch.zeros(1, 64, 1, 32)
    a_log = torch.tensor([math.log(math.log(2.0))])
    dt_bias = torch.full((1, 32), math.log(math.expm1(1.0)))

    spans = gate_tile_spans(raw_decay, a_log, dt_bias)

    assert spans.shape == (4,)
    torch.testing.assert_close(spans, torch.full((4,), 15.0))
