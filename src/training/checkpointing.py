from pathlib import Path

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    extra: dict | None = None,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "extra": extra if extra is not None else {},
    }
    torch.save(payload, str(p))


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int, dict]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])

    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    epoch = int(payload.get("epoch", 0))
    step = int(payload.get("step", 0))
    extra = payload.get("extra", {})
    return epoch, step, extra


def export_torchscript(
    path: str,
    model: torch.nn.Module,
) -> None:
    """
    Saves a TorchScript module that accepts [B,1,F,T] and returns [B,S,F,T].
    Uses scripting (not tracing) so T can vary.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu").eval()
    scripted = torch.jit.script(model_cpu)
    scripted.save(str(p))
