import torch

from k3mini.config import load_config
from k3mini.model import K3MiniForCausalLM
from k3mini.training import build_optimizer


def main() -> None:
    torch.manual_seed(0)
    model_cfg, _, train_cfg = load_config("configs/tiny.json")
    model = K3MiniForCausalLM(model_cfg)
    optimizer = build_optimizer(model, train_cfg)
    input_ids = torch.randint(0, model_cfg.vocab_size, (2, 24))
    labels = torch.randint(0, model_cfg.vocab_size, (2, 24))
    output = model(
        input_ids,
        labels,
        return_logits=True,
        return_diagnostics=True,
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.logits is not None
    output.loss.backward()
    optimizer.step()
    assert output.logits.shape == (2, 24, model_cfg.vocab_size)
    print(f"smoke test passed: loss={output.loss.item():.4f}")
    print(f"logits={tuple(output.logits.shape)}")
    print(f"params={model.parameter_counts()}")
    print(f"backend={model.backend.as_dict()}")


if __name__ == "__main__":
    main()
