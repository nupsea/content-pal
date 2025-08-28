#!/usr/bin/env python3
from __future__ import annotations
import os, json, random
from typing import List, cast
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder, InputExample, losses
from torch.utils.data import DataLoader, random_split, Dataset

DATA = os.getenv("CE_DATA", "ce_train_pairs.jsonl")
OUT  = os.getenv("CE_OUT",  "ce_netflix")
BASE = os.getenv("CE_BASE", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EPOCHS = int(os.getenv("CE_EPOCHS", "3"))
BSZ    = int(os.getenv("CE_BATCH", "32"))
SEED   = int(os.getenv("SEED", "42"))

def load_pairs(path: str) -> List[InputExample]:
    ex: List[InputExample] = []
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            q, d, y = row["q"], row["d"], float(row["label"])
            ex.append(InputExample(texts=[q, d], label=y))
    return ex

def main():
    random.seed(SEED)
    data = load_pairs(DATA)
    random.shuffle(data)

    # 90/10 split for quick sanity
    n = len(data)
    n_train = max(1, int(n * 0.9))
    train_data, dev_data = random_split(
        cast(Dataset, data), [n_train, n - n_train], generator=torch.Generator().manual_seed(SEED)
    )   
    
    model = CrossEncoder(BASE, num_labels=1)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BSZ)
    dev_loader   = DataLoader(dev_data,   shuffle=False, batch_size=BSZ)

    model.fit(
        train_dataloader=train_loader,
        evaluator=None,                # keep it simple; you can add CE evaluator later
        epochs=EPOCHS,
        warmup_steps=100,
        output_path=OUT
    )
    print(f"saved CE to: {OUT}")

if __name__ == "__main__":
    import torch  # needed for random_split generator above
    main()
