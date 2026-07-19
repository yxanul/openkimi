from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol

import torch

from .config import OptimizerType, TrainConfig


def _disable_variable_length_gns_helper_compile() -> None:
    """Keep GNS kernels compiled while avoiding list-cardinality graph explosions."""
    import importlib

    module = importlib.import_module("gram_newton_schulz.muon.muon")
    helper = module.muon_update_pre_orthogonalize
    original = getattr(helper, "_torchdynamo_orig_callable", None)
    if original is not None:
        # MoE models have many shape groups with different parameter counts.
        # The upstream fullgraph helper specializes on every Python-list
        # length, reaches Torch's recompile limit, and never reaches the actual
        # Hopper GNS kernels. The helper is only momentum foreach arithmetic;
        # the Gram-Newton-Schulz operator and its CUDA kernels remain unchanged.
        module.muon_update_pre_orthogonalize = torch.compiler.disable(original)


class OptimizerLike(Protocol):
    @property
    def param_groups(self) -> list[dict[str, Any]]: ...

    def step(self, closure: Any = None) -> Any: ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None: ...


def _named_trainable_parameters(
    model: torch.nn.Module,
) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            yield name, parameter


def _uses_muon(
    name: str,
    parameter: torch.nn.Parameter,
    embedding_parameter_ids: set[int],
) -> bool:
    del name
    return parameter.ndim >= 2 and id(parameter) not in embedding_parameter_ids


def partition_muon_parameters(
    model: torch.nn.Module,
) -> tuple[
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
    list[torch.nn.Parameter],
]:
    """Partition matrix weights for Muon and the remaining AdamW parameters."""
    embedding_parameter_ids: set[int] = set()
    for module_name in ("token_embedding", "lm_head"):
        module = getattr(model, module_name, None)
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            embedding_parameter_ids.add(id(weight))

    muon: list[torch.nn.Parameter] = []
    adam_decay: list[torch.nn.Parameter] = []
    adam_no_decay: list[torch.nn.Parameter] = []
    no_decay_fragments = ("norm", "bias", "A_log", "dt_bias", "correction")
    for name, parameter in _named_trainable_parameters(model):
        if _uses_muon(name, parameter, embedding_parameter_ids):
            muon.append(parameter)
        elif parameter.ndim < 2 or any(
            fragment in name for fragment in no_decay_fragments
        ):
            adam_no_decay.append(parameter)
        else:
            adam_decay.append(parameter)

    partitioned = {id(parameter) for parameter in (*muon, *adam_decay, *adam_no_decay)}
    expected = {id(parameter) for _, parameter in _named_trainable_parameters(model)}
    if partitioned != expected:
        raise RuntimeError("optimizer parameter partition is incomplete or duplicated")
    return muon, adam_decay, adam_no_decay


class MuonClipOptimizer:
    """Dao-AILab Muon plus AdamW scalar updates and Moonshot QK-Clip."""

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: TrainConfig,
        *,
        muon_parameters: list[torch.nn.Parameter],
        adam_decay: list[torch.nn.Parameter],
        adam_no_decay: list[torch.nn.Parameter],
    ) -> None:
        try:
            from gram_newton_schulz import Muon
        except ImportError as error:
            raise RuntimeError(
                "MuonClip requires the pinned Gram Newton-Schulz package; "
                "run scripts/install_sonic_isolated.sh and launch through "
                "scripts/run_with_sonic.sh"
            ) from error
        _disable_variable_length_gns_helper_compile()

        self.model = model
        self.cfg = cfg
        self.muon_parameters = list(muon_parameters)
        self.scalar_optimizer = torch.optim.AdamW(
            [
                {
                    "params": adam_decay,
                    "weight_decay": cfg.weight_decay,
                    "optimizer_kind": "adamw",
                },
                {
                    "params": adam_no_decay,
                    "weight_decay": 0.0,
                    "optimizer_kind": "adamw",
                },
            ],
            lr=cfg.learning_rate,
            betas=cfg.betas,
            eps=cfg.adam_epsilon,
        )
        self.inner = Muon(
            params=[
                {
                    "params": self.muon_parameters,
                    "lr": cfg.muon_learning_rate,
                    "weight_decay": cfg.weight_decay,
                    "momentum": cfg.muon_momentum,
                    "nesterov": cfg.muon_nesterov,
                    "adjust_lr": "rms_norm",
                    "optimizer_kind": "muon",
                }
            ],
            scalar_optimizer=self.scalar_optimizer,
            lr=cfg.muon_learning_rate,
            weight_decay=cfg.weight_decay,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            adjust_lr="rms_norm",
            ns_algorithm="gram_newton_schulz",
            ns_use_kernels=True,
            ns_coefficients_preset="YOU_COEFFICIENTS",
            ns_max_batch_size=cfg.muon_ns_max_batch_size,
            gram_newton_schulz_restart_iterations=[2],
        )
        self.last_clip_diagnostics: dict[str, float | int] = {
            "maximum_attention_logit": 0.0,
            "clipped_heads": 0,
            "minimum_scale": 1.0,
        }

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.inner.param_groups

    def step(self, closure: Any = None) -> Any:
        loss = self.inner.step(closure)
        apply_qk_clip = getattr(self.model, "apply_qk_clip", None)
        if apply_qk_clip is None:
            raise RuntimeError("MuonClip model does not expose apply_qk_clip")
        self.last_clip_diagnostics = apply_qk_clip(self.cfg.qk_clip_threshold)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def diagnostics(self) -> dict[str, float | int]:
        return dict(self.last_clip_diagnostics)

    def state_dict(self) -> dict[str, Any]:
        muon_state: list[dict[str, Any]] = []
        for parameter in self.muon_parameters:
            state = self.inner.state.get(parameter, {})
            muon_state.append(
                {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in state.items()
                }
            )
        muon_groups = [
            {
                key: value
                for key, value in group.items()
                if key != "params" and not callable(value)
            }
            for group in self.inner._muon_param_groups
        ]
        return {
            "format_version": 1,
            "muon_state": muon_state,
            "muon_groups": muon_groups,
            "scalar_optimizer": self.scalar_optimizer.state_dict(),
            "last_clip_diagnostics": self.last_clip_diagnostics,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if int(state_dict.get("format_version", 0)) != 1:
            raise ValueError("unsupported MuonClip optimizer checkpoint format")
        saved_muon_state = list(state_dict["muon_state"])
        if len(saved_muon_state) != len(self.muon_parameters):
            raise ValueError(
                "Muon parameter count changed across checkpoint resume: "
                f"{len(saved_muon_state)} != {len(self.muon_parameters)}"
            )
        self.inner.state.clear()
        for parameter, saved in zip(
            self.muon_parameters,
            saved_muon_state,
            strict=True,
        ):
            self.inner.state[parameter] = {
                key: value.to(device=parameter.device, dtype=parameter.dtype)
                if torch.is_tensor(value)
                else value
                for key, value in saved.items()
            }
        for group, saved in zip(
            self.inner._muon_param_groups,
            state_dict["muon_groups"],
            strict=True,
        ):
            group.update(saved)
        self.scalar_optimizer.load_state_dict(state_dict["scalar_optimizer"])
        self.last_clip_diagnostics = dict(
            state_dict.get("last_clip_diagnostics", self.last_clip_diagnostics)
        )


def build_optimizer(
    model: torch.nn.Module,
    cfg: TrainConfig,
) -> OptimizerLike:
    if cfg.optimizer is OptimizerType.MUONCLIP:
        muon, adam_decay, adam_no_decay = partition_muon_parameters(model)
        if not muon:
            raise ValueError("MuonClip requires at least one non-embedding matrix")
        configure_qk_clip = getattr(model, "configure_qk_clip", None)
        if configure_qk_clip is None:
            raise RuntimeError("MuonClip model does not expose configure_qk_clip")
        configure_qk_clip(
            enabled=True,
            query_chunk_size=cfg.qk_clip_query_chunk_size,
        )
        return MuonClipOptimizer(
            model,
            cfg,
            muon_parameters=muon,
            adam_decay=adam_decay,
            adam_no_decay=adam_no_decay,
        )

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    no_decay_fragments = ("norm", "bias", "A_log", "dt_bias", "correction")
    for name, parameter in _named_trainable_parameters(model):
        if parameter.ndim < 2 or any(
            fragment in name for fragment in no_decay_fragments
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": cfg.weight_decay,
                "optimizer_kind": "adamw",
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "optimizer_kind": "adamw",
            },
        ],
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.adam_epsilon,
    )
