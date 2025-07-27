from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class CausalLMDataset(Dataset):
    """Minimal dataset that tokenises text for causal language modelling.

    Each example returns *input_ids* and *labels* where *labels* are a
    direct copy of *input_ids*. Tokens that should be ignored by the
    loss function can be set to ``-100`` if you add padding later.
    """

    def __init__(self, texts: List[str], tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.examples = tokenizer(texts, truncation=True, max_length=max_length, add_special_tokens=True)

    def __len__(self):
        return len(self.examples["input_ids"])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples["input_ids"][idx], dtype=torch.long)
        # For language modelling the labels are the same as the inputs.
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}


def get_loader(
    model_name: str,
    split: str = "validation",
    batch_size: int = 8,
    max_length: int = 512,
    num_workers: int = 2,
):
    """Return a DataLoader of tokenised text from the Wikitext-103 dataset."""

    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenised_ds = CausalLMDataset(dataset["text"], tokenizer, max_length=max_length)

    def collate(batch):
        # Pad to longest sequence in batch.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in batch],
            batch_first=True,
            padding_value=-100,  # HF ignores -100 in loss computation.
        )
        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(
        tokenised_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
