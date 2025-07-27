import csv, math, copy, torch
from huggingface_hub import hf_hub_download

from analysis.evaluate import evaluate_loss, estimate_llc_for_ckpt   # just created!
# (build `model`, `tokenizer`, `valid_loader` exactly as you did before)
from data import get_dataloaders


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------------
# build model exactly the same config you used earlier
from transformer_lens import HookedTransformer, HookedTransformerConfig
cfg = HookedTransformerConfig(
    n_layers=2, d_model=256, n_ctx=1024,
    d_head=256 // 8, n_heads=8, d_vocab=5000,
    attn_only=True, normalization_type="LN",
    positional_embedding_type="shortformer",
    tokenizer_name='georgeyw/TinyStories-tokenizer-5k'
    train_loader, val_loader = get_dataloaders(tokenizer)
    n_tokens = _SEQ_LEN * len(val_loader.dataset)   # ← needed for β or logging
)
model = HookedTransformer(cfg).to(device).eval()
# -------------------------------------------------------------------------

# Your DataLoader: use "test" split here
test_loader = valid_loader   # (rename if you loaded a proper test set)

# convenience
seq_len = test_loader.dataset.seqs[0].__len__()
n_tokens = len(test_loader.dataset) * (seq_len - 1)
beta = 1.0 / math.log(n_tokens)

steps = list(range(500, 20001, 500))  # 500 ... 20000
epsilon = 1e-3                        # perturbation scale

results = []

for step in steps:
    ckpt_path = hf_hub_download(
        "timaeus/tetrahedron-3m-og",
        f"checkpoints/checkpoint_{step:07d}.pth"
    )

    # ---- LLC estimation --------------------------------------------------
    base_loss, E_beta_loss, llc_hat = estimate_llc_for_ckpt(
        model, ckpt_path, val_loader,
        beta=beta,
        gamma=1e4,
        sgld_steps=200,
        sgld_lr=1e-4,
        device=device
        train_loader=train_loader
    )

    # ---- ε-perturbation ---------------------------------------------------
    noise = {n: torch.randn_like(p) for n, p in model.named_parameters()}
    original_state = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(epsilon * noise[name])
    perturbed_loss = evaluate_loss(model, test_loader, device)
    delta_L = perturbed_loss - base_loss
    model.load_state_dict(original_state)

    results.append((step, base_loss, E_beta_loss, llc_hat,
                    perturbed_loss, delta_L))
    print(f"step {step:5}: L={base_loss:.4f}  λ̂={llc_hat:8.1f}  ΔL={delta_L:+.5f}")

# -------------------------------------------------------------------------
# save as CSV
with open("llc_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["step", "baseline_loss", "E_beta_loss",
                     "llc_hat", "perturbed_loss", "delta_L"])
    writer.writerows(results)

print("✓ Saved → llc_results.csv")
