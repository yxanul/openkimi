from __future__ import annotations

from types import ModuleType

import pytest

from k3mini.backends import _validated_kda_backend_name


def _patched_fla(
    *,
    version: int = 1,
    span: float = 232.0,
    precision: str = "tf32x3",
) -> ModuleType:
    module = ModuleType("chunk_intra")
    module.OPENKIMI_KDA_GUARDED_DIAGONAL_VERSION = version
    module._OPENKIMI_GUARD_MAX_LOG2_SPAN = span
    module._OPENKIMI_GUARD_DOT_PRECISION = precision
    return module


def test_validated_kda_backend_reports_pinned_patch() -> None:
    name = _validated_kda_backend_name(_patched_fla())

    assert name == (
        "fla.ops.kda.chunk_kda("
        "openkimi_guarded_diagonal_v1,tf32x3,span<=232)"
    )


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (_patched_fla(version=0), "guarded KDA patch v1"),
        (_patched_fla(span=240.0), "exceeds the validated maximum 232"),
        (_patched_fla(precision="tf32"), "is not validated"),
    ],
)
def test_validated_kda_backend_rejects_unvalidated_fla(
    module: ModuleType,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _validated_kda_backend_name(module)
