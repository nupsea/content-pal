#!/usr/bin/env python3
from __future__ import annotations
import os, math
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

CSV_IN  = os.getenv("CSV_IN",  "../../data/netflix_titles_cleaned.csv")
CSV_OUT = os.getenv("CSV_OUT", "../../data/netflix_titles_expanded.csv")
MODEL   = os.getenv("D2Q_MODEL", "doc2query/msmarco-t5-small-v1")   # or "doc2query/msmarco-t5-base-v1"
N_QUERIES = int(os.getenv("D2Q_N", "6"))                             # 4–8 is typical
BATCH      = int(os.getenv("D2Q_BATCH", "8"))                        # 8–16 on CPU; more if GPU

def main():
    df = pd.read_csv(CSV_IN)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # compact, high-signal input per row
    def prep(row):
        title = (row.get("title") or "").strip()
        desc  = (row.get("description") or "").strip()
        text  = (title + ". " + desc)[:512]
        return text or title

    inputs = df.apply(prep, axis=1).tolist()
    expanded = []

    for i in tqdm(range(0, len(inputs), BATCH), desc="doc2query"):
        batch = inputs[i:i+BATCH]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                num_beams=max(4, N_QUERIES),
                num_return_sequences=N_QUERIES,
                max_length=48,
                early_stopping=True,
                do_sample=False,
            )
        # group N_QUERIES per source
        decoded = [tok.decode(o, skip_special_tokens=True) for o in out]
        for j in range(len(batch)):
            qs = decoded[j*N_QUERIES:(j+1)*N_QUERIES]
            # join with spaces so BM25 treats them as normal text
            expanded.append(" ".join(qs))

    df["expanded_text"] = expanded
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT}   (rows={len(df)})")

if __name__ == "__main__":
    main()
