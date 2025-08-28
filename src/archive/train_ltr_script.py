import json
from asset_ltr import train_ltr, save_model, RET
from open_search import Cfg, make_client

cfg = Cfg()
os_client = make_client(cfg)
INDEX = "netflix_assets"

# Your eval/train pairs: {"s123": ["kids series", "cat cartoon", ...], ...}
with open("ground_truth.json","r") as f:
    qid_to_queries = json.load(f)

model = train_ltr(os_client, INDEX, qid_to_queries, cfg=RET)
save_model(model, "asset_ltr.bin")
print("Saved LTR model -> asset_ltr.bin")
