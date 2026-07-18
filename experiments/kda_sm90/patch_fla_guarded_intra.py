"""Apply or restore the pinned FLA guarded-diagonal integration experiment.

This deliberately patches only the isolated H100 environment's installed FLA
source. The source hash and exact structural markers are checked before any
write so a changed upstream revision fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import textwrap
from pathlib import Path

EXPECTED_SHA256 = "09979c0d1e768ce7efd34fbba8b6ce019b542f21e62c13f7a1f79bee3d0a9a0c"
PATCH_MARKER = "OPENKIMI_GUARDED_DIAGONAL"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_path() -> Path:
    spec = importlib.util.find_spec("fla.ops.kda.chunk_intra")
    if spec is None or spec.origin is None:
        raise RuntimeError("could not locate fla.ops.kda.chunk_intra")
    return Path(spec.origin)


def _guard_code() -> str:
    return textwrap.dedent(
        """\
        # OPENKIMI_GUARDED_DIAGONAL: midpoint factorization with an exact fallback.
        b_guard_rows = o_i[:, None]
        m_guard = ((i_ti + b_guard_rows) < T) & m_k[None, :]
        b_guard_max = tl.max(tl.where(m_guard, b_g, -3.402823e38), axis=0)
        b_guard_min = tl.min(tl.where(m_guard, b_g, 3.402823e38), axis=0)
        b_guard_span = tl.max(
            tl.where(m_k, b_guard_max - b_guard_min, 0.0),
            axis=0,
        )
        b_guard_previous = tl.load(
            g + (i_ti + b_guard_rows - 1) * HV*K + o_k[None, :],
            mask=(b_guard_rows > 0) & m_guard,
            other=0.0,
        ).to(tl.float32)
        b_guard_monotonic = (
            (b_guard_rows == 0) | ~m_guard | (b_g <= b_guard_previous)
        )
        b_guard_finite = (
            ~m_guard | ((b_g == b_g) & (tl.abs(b_g) < 3.402823e38))
        )
        b_guard_valid_count = tl.sum(
            tl.sum((b_guard_monotonic & b_guard_finite).to(tl.int32), axis=1),
            axis=0,
        )
        b_guard_fast = (
            (b_guard_valid_count == BC * BK)
            & (b_guard_span <= GUARD_MAX_LOG2_SPAN)
        )
        b_guard_ref = ((b_guard_max + b_guard_min) * 0.5)[None, :]
        """
    )


def _first_fast_path() -> str:
    return textwrap.dedent(
        """\
        p_dAqk = tl.make_block_ptr(
            dAqk, (T, BT), (HV*BT, 1),
            (i_ti, i_i * BC), (BC, BC), (1, 0),
        )
        p_dAkk = tl.make_block_ptr(
            dAkk, (T, BT), (HV*BT, 1),
            (i_ti, i_i * BC), (BC, BC), (1, 0),
        )
        b_dAqk_diag_qk = tl.load(
            p_dAqk, boundary_check=(0, 1)
        ).to(tl.float32)
        b_dAkk_diag_qk = tl.load(
            p_dAkk, boundary_check=(0, 1)
        ).to(tl.float32)
        m_i_diag_qk = (
            (o_i[:, None] >= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_diag_qk = (i_ti + o_i[:, None]) < T
        b_dAqk_diag_qk = tl.where(
            m_i_diag_qk, b_dAqk_diag_qk, 0.0
        )
        b_dAkk_diag_qk = tl.where(
            m_i_diag_qk, b_dAkk_diag_qk, 0.0
        )
        b_g_diag_qk = tl.where(
            m_j_diag_qk, b_g - b_guard_ref, 0.0
        )
        exp_b_g_diag_qk = tl.where(
            m_j_diag_qk, exp2(b_g_diag_qk), 0.0
        )
        exp_neg_b_g_diag_qk = tl.where(
            m_j_diag_qk, exp2(-b_g_diag_qk), 0.0
        )
        b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
        b_dq2 += (
            tl.dot(
                b_dAqk_diag_qk,
                b_k_exp_diag_qk,
                input_precision=GUARD_DOT_PRECISION,
            )
            * exp_b_g_diag_qk
        )
        b_dk2 += (
            tl.dot(
                b_dAkk_diag_qk,
                b_k_exp_diag_qk,
                input_precision=GUARD_DOT_PRECISION,
            )
            * exp_b_g_diag_qk
        )
        """
    )


def _second_fast_path() -> str:
    return textwrap.dedent(
        """\
        p_q = tl.make_block_ptr(
            q, (T, K), (H*K, 1),
            (i_ti, i_k * BK), (BC, BK), (1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        p_b = tl.make_block_ptr(
            beta, (T,), (HV,), (i_ti,), (BC,), (0,),
        )
        b_b = tl.load(p_b, boundary_check=(0,))
        p_dAqk = tl.make_block_ptr(
            dAqk, (BT, T), (1, HV*BT),
            (i_i * BC, i_ti), (BC, BC), (0, 1),
        )
        p_dAkk = tl.make_block_ptr(
            dAkk, (BT, T), (1, HV*BT),
            (i_i * BC, i_ti), (BC, BC), (0, 1),
        )
        b_dAqk_diag_kk = tl.load(
            p_dAqk, boundary_check=(0, 1)
        ).to(tl.float32)
        b_dAkk_diag_kk = tl.load(
            p_dAkk, boundary_check=(0, 1)
        ).to(tl.float32)
        m_i_diag_kk = (
            (o_i[:, None] <= o_i[None, :])
            & ((i_ti + o_i[:, None]) < T)
            & ((i_ti + o_i[None, :]) < T)
        )
        m_j_diag_kk = (i_ti + o_i[:, None]) < T
        b_dAqk_diag_kk = tl.where(
            m_i_diag_kk, b_dAqk_diag_kk, 0.0
        )
        b_dAkk_diag_kk = tl.where(
            m_i_diag_kk, b_dAkk_diag_kk, 0.0
        )
        b_g_diag_kk = tl.where(
            m_j_diag_kk, b_g - b_guard_ref, 0.0
        )
        exp_b_g_diag_kk = tl.where(
            m_j_diag_kk, exp2(b_g_diag_kk), 0.0
        )
        exp_neg_b_g_diag_kk = tl.where(
            m_j_diag_kk, exp2(-b_g_diag_kk), 0.0
        )
        b_q_exp = b_q * exp_b_g_diag_kk
        b_kb_exp = b_k * b_b[:, None] * exp_b_g_diag_kk
        b_dkt += (
            tl.dot(
                b_dAqk_diag_kk,
                b_q_exp,
                input_precision=GUARD_DOT_PRECISION,
            )
            * exp_neg_b_g_diag_kk
        )
        b_dkt += (
            tl.dot(
                b_dAkk_diag_kk,
                b_kb_exp,
                input_precision=GUARD_DOT_PRECISION,
            )
            * exp_neg_b_g_diag_kk
        )
        """
    )


def _replace_else_region(
    source: str,
    *,
    start_marker: str,
    end_marker: str,
    fast_path: str,
) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    fallback = source[start + len("    else:\n") : end]
    replacement = (
        "    else:\n"
        + textwrap.indent(_guard_code(), "        ")
        + "        if b_guard_fast:\n"
        + textwrap.indent(fast_path, "            ")
        + "        else:\n"
        + textwrap.indent(fallback, "    ")
    )
    return source[:start] + replacement + source[end:]


def _patched_source(source: str) -> str:
    import_marker = "import torch\nimport triton\n"
    guarded_import = (
        "import os  # OPENKIMI_GUARDED_DIAGONAL\n"
        "import torch\n"
        "import triton\n"
    )
    if source.count(import_marker) != 1:
        raise RuntimeError("unexpected FLA import block")
    source = source.replace(import_marker, guarded_import, 1)

    config_marker = "if IS_TF32_SUPPORTED:\n"
    guarded_config = (
        "OPENKIMI_KDA_GUARDED_DIAGONAL_VERSION = 1\n"
        "_OPENKIMI_GUARD_MAX_LOG2_SPAN = float(\n"
        '    os.environ.get("K3MINI_KDA_GUARD_SPAN", "232.0")\n'
        ")\n"
        "_OPENKIMI_GUARD_DOT_PRECISION = os.environ.get(\n"
        '    "K3MINI_KDA_GUARD_DOT_PRECISION", "tf32x3"\n'
        ")\n"
        "if _OPENKIMI_GUARD_DOT_PRECISION not in "
        "{\"tf32\", \"tf32x3\", \"ieee\"}:\n"
        "    raise ValueError(\n"
        '        "K3MINI_KDA_GUARD_DOT_PRECISION must be tf32, tf32x3, or ieee"\n'
        "    )\n"
        "\n"
        "if IS_TF32_SUPPORTED:\n"
    )
    if source.count(config_marker) != 1:
        raise RuntimeError("unexpected FLA precision configuration")
    source = source.replace(config_marker, guarded_config, 1)

    jit_marker = (
        "@triton.jit(do_not_specialize=['B', 'T'])\n"
        "def chunk_kda_bwd_kernel_intra("
    )
    guarded_jit = (
        "@triton.jit(\n"
        "    do_not_specialize=['B', 'T', 'GUARD_MAX_LOG2_SPAN']\n"
        ")\n"
        "def chunk_kda_bwd_kernel_intra("
    )
    if source.count(jit_marker) != 1:
        raise RuntimeError("unexpected FLA backward-intra JIT decorator")
    source = source.replace(jit_marker, guarded_jit, 1)

    signature = (
        "    SAFE_GATE: tl.constexpr,\n"
        "    USE_GATHER: tl.constexpr,\n"
        "):"
    )
    guarded_signature = (
        "    SAFE_GATE: tl.constexpr,\n"
        "    USE_GATHER: tl.constexpr,\n"
        "    GUARD_MAX_LOG2_SPAN,\n"
        "    GUARD_DOT_PRECISION: tl.constexpr,  # OPENKIMI_GUARDED_DIAGONAL\n"
        "):"
    )
    if source.count(signature) != 1:
        raise RuntimeError("unexpected FLA backward-intra signature")
    source = source.replace(signature, guarded_signature, 1)

    first_start = (
        "    else:\n"
        "        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):\n"
        "            # [BC]\n"
        "            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)\n"
    )
    source = _replace_else_region(
        source,
        start_marker=first_start,
        end_marker="\n\n    b_db = tl.sum(b_dk2 * b_k, 1)",
        fast_path=_first_fast_path(),
    )

    second_start = (
        "    else:\n"
        "        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):\n"
        "            # [BC,]\n"
        "            b_dAqk = tl.load(dAqk + o_dA + j * HV*BT)\n"
    )
    source = _replace_else_region(
        source,
        start_marker=second_start,
        end_marker="\n    p_dk = tl.make_block_ptr(",
        fast_path=_second_fast_path(),
    )

    launch = (
        "        SAFE_GATE=safe_gate,\n"
        "        USE_GATHER=IS_GATHER_SUPPORTED,\n"
        "    )"
    )
    guarded_launch = (
        "        SAFE_GATE=safe_gate,\n"
        "        USE_GATHER=IS_GATHER_SUPPORTED,\n"
        "        GUARD_MAX_LOG2_SPAN=_OPENKIMI_GUARD_MAX_LOG2_SPAN,\n"
        "        GUARD_DOT_PRECISION=_OPENKIMI_GUARD_DOT_PRECISION,\n"
        "    )"
    )
    if source.count(launch) != 1:
        raise RuntimeError("unexpected FLA backward-intra launch")
    return source.replace(launch, guarded_launch, 1)


def _clear_bytecode(module: Path) -> None:
    cache = module.parent / "__pycache__"
    if cache.exists():
        for path in cache.glob("chunk_intra.*.pyc"):
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument(
        "--module-path",
        type=Path,
        help="patch an explicit FLA source-tree file instead of the installed module",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not create the .openkimi-exact backup when patching a source tree",
    )
    args = parser.parse_args()
    if args.restore and args.no_backup:
        raise ValueError("--restore and --no-backup cannot be combined")
    module = args.module_path if args.module_path is not None else _module_path()
    backup = module.with_suffix(".py.openkimi-exact")

    if args.restore:
        if not backup.exists():
            raise RuntimeError(f"backup does not exist: {backup}")
        shutil.copy2(backup, module)
        _clear_bytecode(module)
        print(f"restored {module} ({_sha256(module)})")
        return

    source = module.read_text()
    if PATCH_MARKER in source:
        print(f"guarded diagonal is already applied to {module}")
        return
    actual_hash = _sha256(module)
    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            f"refusing to patch unexpected FLA source: {actual_hash}"
        )
    if not args.no_backup:
        shutil.copy2(module, backup)
    module.write_text(_patched_source(source))
    _clear_bytecode(module)
    if args.no_backup:
        print(f"patched {module} without a backup")
    else:
        print(f"patched {module}; exact backup is {backup}")


if __name__ == "__main__":
    main()
