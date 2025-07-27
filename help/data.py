# Cell 2 — Build model & val_loader
CFG = HookedTransformerConfig(
    n_layers=2, n_heads=8, d_head=32, d_model=256,
    n_ctx=1024, d_vocab=5000, attn_only=True,
    tokenizer_name="georgeyw/TinyStories-tokenizer-5k",
    normalization_type="LN",
    positional_embedding_type="shortformer"
)
model = HookedTransformer(CFG).eval()

class PileValidDataset(Dataset):
    def __init__(self, lines, seq_len=1024):
        toks=[]
        for ln in lines:
            toks.extend(model.tokenizer(ln, truncation=False)["input_ids"])
        self.seqs=[toks[i:i+seq_len] for i in range(0,len(toks),seq_len)]
        if len(self.seqs[-1])<seq_len: self.seqs.pop()
    def __len__(self): return len(self.seqs)
    def __getitem__(self,i):
        x=torch.tensor(self.seqs[i][:-1],dtype=torch.long)
        y=torch.tensor(self.seqs[i][1:], dtype=torch.long)
        return {"input_ids":x, "labels":y}

# load 512 lines from Pile...
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json
lines=[]
path = hf_hub_download("stanford-crfm/DSIR-filtered-pile-50M","train_6.jsonl",repo_type="dataset")
with open(path) as f:
    for _ in range(512):
        lines.append(json.loads(next(f))["contents"])
val_ds = PileValidDataset(lines, seq_len=1024)
val_loader=DataLoader(val_ds, batch_size=16, shuffle=False)
