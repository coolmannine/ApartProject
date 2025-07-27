import torch
from torch.nn import functional as F
from typing import Dict


def evaluate_loss(model, data_loader, device: str = "cpu") -> float:
    """Compute average token-level cross-entropy loss over *data_loader*.

    Assumes each batch from *data_loader* is a dict with at least
    ``input_ids`` and ``labels`` keys and that ``labels`` already has
    the special value ``-100`` where tokens should be ignored (HF
    convention).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # HuggingFace models support passing labels directly.
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss: torch.Tensor = outputs.loss

            # Count non-ignored tokens so we can return bits-per-token.
            valid_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return total_loss / total_tokens if total_tokens > 0 else float("nan") 