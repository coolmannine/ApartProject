
import csv, math, copy, torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig

from analysis.data import get_dataloaders, _SEQ_LEN     # same folder
from analysis.evaluate import evaluate_loss, estimate_llc_for_ckpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- model --------------------------------------------------------------
cfg = HookedTransformerConfig(
    n_layers=2, d_model=256, n_ctx=_SEQ_LEN,
    d_head=256//8, n_heads=8, d_vocab=5000,
    attn_only=True, normalization_type="LN",
    positional_embedding_type="shortformer",
    tokenizer_name='georgeyw/TinyStories-tokenizer-5k'
)
model = HookedTransformer(cfg).to(device)

# ----- data ---------------------------------------------------------------
train_loader, val_loader, tokenizer = get_dataloaders()
seq_len = _SEQ_LEN
n_tokens = len(val_loader.dataset) * (seq_len-1)
beta = 1.0 / math.log(n_tokens)

# ----- sweep checkpoints --------------------------------------------------
epsilon = 1e-3
steps   = list(range(500, 20001, 500))
results = []

for step in steps:
    ckpt_path = hf_hub_download(
        "timaeus/tetrahedron-3m-og",
        f"checkpoints/checkpoint_{step:07d}.pth"
    )

    # LLC ------------------------------------------------------------------
    base_loss, Ebeta_loss, llc_hat = estimate_llc_for_ckpt(
        model, ckpt_path,
        val_loader,          # acts as “test_loader” inside the fn
        beta=beta, gamma=1e4,
        sgld_steps=200, sgld_lr=1e-4,
        device=device
)

    # ε-perturb ------------------------------------------------------------
    noise = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    orig  = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.add_(epsilon * noise[n])

    pert_loss = evaluate_loss(model, val_loader, device)
    delta_L   = pert_loss - base_loss
    model.load_state_dict(orig)

    # record ---------------------------------------------------------------
    results.append((step, base_loss, Ebeta_loss, llc_hat, pert_loss, delta_L))
    print(f"[{step:05}]  L={base_loss:.4f}  λ̂={llc_hat:8.1f}  ΔL={delta_L:+.5f}")

# ----- save ---------------------------------------------------------------
with open("llc_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step","baseline_loss","Ebeta_loss","llc_hat",
                     "perturbed_loss","delta_L"])
    writer.writerows(results)

print("✓  Saved llc_results.csv")
