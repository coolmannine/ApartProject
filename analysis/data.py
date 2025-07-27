# data.py
import json, torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

_TOKENIZER_NAME = "georgeyw/TinyStories-tokenizer-5k"
_SEQ_LEN        = 1024          # must match model cfg
_BATCH_TRAIN    = 16
_BATCH_VAL      = 16

def _download_lines(num_files=2):
    """Pull a few jsonl shards (~50 k lines each) from the filtered pile."""
    valid_lines = []
    # we just hard-code 2 shards for ≈100 k lines:
    file_names = [f"train_{i}.jsonl" for i in [6, 5]][:num_files]
    for fname in file_names:
        p = hf_hub_download("stanford-crfm/DSIR-filtered-pile-50M",
                            fname, repo_type="dataset")
        with open(p, "r") as f:
            for ln in f:
                valid_lines.append(json.loads(ln)["contents"])
    return valid_lines

class PileDataset(Dataset):
    def __init__(self, lines, tokenizer, seq_len=_SEQ_LEN):
        toks = []
        for ln in lines:
            toks.extend(tokenizer(ln, truncation=False)["input_ids"])
        # split into fixed-length chunks
        self.seqs = [toks[i:i+seq_len]
                     for i in range(0, len(toks), seq_len)]
        if len(self.seqs[-1]) < seq_len: self.seqs.pop()

    def __len__(self):  return len(self.seqs)
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx][:-1])
        y = torch.tensor(self.seqs[idx][1:])
        return dict(input_ids=x, labels=y)

def get_dataloaders():
    tok = PreTrainedTokenizerFast.from_pretrained(_TOKENIZER_NAME)
    tok.model_max_length = _SEQ_LEN

    lines = _download_lines()          # ~100 k lines
    split = int(0.95 * len(lines))     # 95 k / 5 k
    train_lines, val_lines = lines[:split], lines[split:]

    train_ds = PileDataset(train_lines, tok)
    val_ds   = PileDataset(val_lines, tok)

    return (DataLoader(train_ds, batch_size=_BATCH_TRAIN, shuffle=True),
            DataLoader(val_ds,   batch_size=_BATCH_VAL,   shuffle=False),
            tok)

__all__ = ["get_dataloaders", "_SEQ_LEN"]
