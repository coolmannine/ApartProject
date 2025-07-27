"""
evaluate.py
--------------------------------------------------------------------
• evaluate_loss             – compute full-set CE loss (tokens / nats)
• ExpectationEstimator      – running E[·] helper  (copied from repo)
• estimate_llc_for_ckpt     – SGLD-based local learning-coefficient
--------------------------------------------------------------------
"""
import math, copy, torch, torch.nn.functional as F
from analysis.estimator import ExpectationEstimator        # your estimator.py

# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_loss(model, dataloader, device="cuda"):
    """Cross-entropy per token on an entire DataLoader."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in dataloader:
        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )
        total_loss  += loss.item()
        total_tokens += y.numel()
    return total_loss / total_tokens
# ------------------------------------------------------------------


def _quadratic_pull(model, w_star_state, gamma: float):
    """Return γ·½·||w-w★||² *value*  and the per-param grads (p – p★)."""
    quad, grads = 0., []
    for p, p_star in zip(model.parameters(), w_star_state.values()):
        diff = p - p_star
        quad += diff.pow(2).sum()
        grads.append(diff)                    # gradient of ½‖·‖² wrt p
    return gamma * 0.5 * quad, grads
# ------------------------------------------------------------------


def estimate_llc_for_ckpt(model,
                          ckpt_path          : str,
                          test_loader        ,              # DataLoader
                          beta               : float,       # 1 / log(n_tokens)
                          gamma              : float = 1e4,
                          sgld_steps         : int   = 200,
                          sgld_lr            : float = 1e-4,
                          device             : str   = "cuda"):
    """
    Computes λ̂ for a single checkpoint.
    Returns:  (baseline_loss, E_beta[loss],  λ̂)
    """
    # ---- (a) load checkpoint  --------------------------------------------
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd.get("model_state_dict", sd))

    # baseline loss at w★
    w_star_loss = evaluate_loss(model, test_loader, device)

    # total #tokens processed when evaluating loss each time
    n_tokens = len(test_loader.dataset) * (test_loader.dataset.seqs[0].__len__() - 1)

    # freeze a copy of w★ for quadratic regulariser
    w_star_state = copy.deepcopy(model.state_dict())

    # ---- (b) estimator over SGLD samples ---------------------------------
    est = ExpectationEstimator(num_samples=sgld_steps,
                               observable_dim=1,
                               device=device)

    # ---- (c) localised tempered SGLD ------------------------------------
    data_iter = iter(test_loader)
    for t in range(sgld_steps):
        model.train()

        # recycle batches
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(test_loader)
            batch = next(data_iter)

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)                    # forward
        loss   = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            reduction="mean"
        )

        quad_val, quad_grads = _quadratic_pull(model, w_star_state, gamma)
        total = beta * n_tokens * loss + quad_val       # scalar

        # back-prop
        total.backward()
        with torch.no_grad():
            for p, qg in zip(model.parameters(), quad_grads):
                if p.grad is None: continue
                p.grad += gamma * qg                    # ∂quad
                p.add_( -sgld_lr * p.grad )             # grad step
                p.add_( torch.randn_like(p) * math.sqrt(2*sgld_lr) )  # noise
            model.zero_grad()

        # evaluate sample loss on full test-set
        l_sample = evaluate_loss(model, test_loader, device)
        est.update(chain=0, draw=t,
                   observation=torch.tensor([l_sample], device=device))

    E_Ln = est.estimate()["mean"].item()

    # ---- (d) λ̂  ----------------------------------------------------------
    llc_hat = n_tokens * beta * (E_Ln - w_star_loss)
    return w_star_loss, E_Ln, llc_hat
# ------------------------------------------------------------------
