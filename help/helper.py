# Cell 3 — Helpers (no prints here)
def ce_per_token(m, loader):
    m.eval()
    tot=0; nt=0
    with torch.no_grad():
        for b in loader:
            x,y=b["input_ids"].to(DEVICE), b["labels"].to(DEVICE)
            logits=m(x)
            l=F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                y.view(-1), reduction="sum"
            )
            tot+=l.item(); nt+=y.numel()
    return tot/nt

def perturb_state(sd, frac=PERT_FRAC):
    out={}
    for k,v in sd.items():
        if torch.is_floating_point(v):
            sigma=frac*v.norm()/math.sqrt(v.numel())
            out[k]=v+torch.randn_like(v)*sigma
        else:
            out[k]=v
    return out

def llc_mean(m):
    stats = estimate_learning_coeff_with_summary(
        m,
        loader           = val_loader,
        evaluate         = ce_per_token,
        sampling_method  = SGLD,
        optimizer_kwargs = dict(
            lr=1e-3,
            localization=200.0,
            nbeta=default_nbeta(val_loader),
        ),
        num_chains=20, num_draws=200,
        num_burnin_steps=0, num_steps_bw_draws=1,
        device=DEVICE, online=False, verbose=False,
    )
    return float(stats["llc/mean"])
