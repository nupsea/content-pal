#!/usr/bin/env python3
from __future__ import annotations
import os, re
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

# =========================
# Config via environment
# =========================
def as_bool(x) -> bool:
    return str(x).strip().lower() in ("1","true","t","yes","y","on")

OS_URL       = os.getenv("OS_URL", "https://localhost:9200")
OS_USER      = os.getenv("OS_USER", "admin")
OS_PWD       = os.getenv("OS_PASS", "admin")
OS_VERIFY    = as_bool(os.getenv("OS_VERIFY", "0"))
OS_CA        = os.getenv("OS_CA")  # optional
INDEX_NAME   = os.getenv("OS_INDEX", "netflix_assets_v6")
CSV_NAME     = os.getenv("CSV_NAME", "netflix_titles_expanded.csv")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_BATCH  = int(os.getenv("EMBED_BATCH", "64"))  # Reduced batch size for larger model

# =========================
# Client
# =========================
def make_client() -> OpenSearch:
    auth = (OS_USER, OS_PWD) if (OS_USER and OS_PWD) else None
    if OS_URL.startswith("https://"):
        return OpenSearch(
            OS_URL,
            http_auth=auth,
            verify_certs=OS_VERIFY,
            ca_certs=OS_CA if OS_VERIFY else None,
            ssl_assert_hostname=OS_VERIFY,
            ssl_show_warn=OS_VERIFY,
            connection_class=RequestsHttpConnection,
            timeout=60, max_retries=3, retry_on_timeout=True,
        )
    return OpenSearch(OS_URL, http_auth=auth)

# =========================
# Mapping (simple & general)
# =========================
def index_mapping(vector_dim: int) -> Dict[str, Any]:
    """
    Keep this simple: standard analyzers, english for description only.
    We rely on PRF (significant_text) for expansion—no synonym files.
    """
    return {
        "settings": {
            "index": {"knn": True, "refresh_interval": "1s"},
            "analysis": {
                "analyzer": {
                    "english_desc": {"tokenizer": "standard", "filter": ["lowercase","porter_stem","stop"]}
                }
            }
        },
        "mappings": {
            "properties": {
                "show_id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "director": {"type": "text"},
                "cast": {"type": "text"},
                "cast_list": {"type": "keyword"},
                "country": {"type": "keyword"},
                "date_added": {"type": "date"},
                "date_added_raw": {"type": "keyword"},
                "release_year": {"type": "integer"},
                "rating": {"type": "keyword"},
                "duration": {"type": "keyword"},
                "listed_in": {"type": "keyword"},
                # text copy for categories (so PRF can mine significant terms)
                "listed_in_text": {"type": "text"},
                "description": {"type": "text", "analyzer": "english_desc"},
                # NEW: offline expansions for BM25
                "expanded_text": {"type": "text"},
                "vector": {
                    "type": "knn_vector",
                    "dimension": vector_dim,
                    "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"},
                },
            }
        }
    }

def recreate_index(client: OpenSearch, index: str, vector_dim: int) -> None:
    try:
        if client.indices.exists(index=index):
            client.indices.delete(index=index)
            print(f"Deleted old index: {index}")
    except Exception as e:
        print(f"Warn: delete {index}: {e}")
    client.indices.create(index=index, body=index_mapping(vector_dim))
    client.cluster.health(index=index)
    client.indices.refresh(index=index)
    print(f"Created index {index}")

# =========================
# Row normalization
# =========================
_DATE_PATTERNS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y")
_RE_WS = re.compile(r"\s+")

def _clean(s: Optional[str]) -> Optional[str]:
    if s is None: return None
    s2 = str(s).strip()
    return s2 or None

def _split_csv(s: Optional[str]) -> List[str]:
    s = _clean(s)
    return [p.strip() for p in s.split(",")] if s else []

def to_iso_date(s: Optional[str]) -> Optional[str]:
    if not s: return None
    s2 = str(s).strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s2, fmt).date().isoformat()
        except Exception:
            continue
    return None

def text_for_embedding(d: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ("title","description","director","type","country","rating"):
        v = d.get(k); parts.append(v) if v else None
    if d.get("cast_list"): parts.append(", ".join(d["cast_list"]))
    if d.get("listed_in"): parts.append(", ".join(d["listed_in"]))
    return " | ".join(parts)

def row_normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    listed = _split_csv(row.get("listed_in"))
    cast_list = _split_csv(row.get("cast"))
    ry = _clean(row.get("release_year"))
    release_year = int(ry) if ry and str(ry).isdigit() else None
    raw_date = _clean(row.get("date_added"))
    iso_date = to_iso_date(raw_date)

    doc: Dict[str, Any] = {
        "show_id": _clean(row.get("show_id")),
        "type": _clean(row.get("type")),
        "title": _clean(row.get("title")),
        "director": _clean(row.get("director")),
        "cast": _clean(row.get("cast")),
        "cast_list": cast_list,
        "country": _clean(row.get("country")),
        "date_added_raw": raw_date,
        **({"date_added": iso_date} if iso_date else {}),
        "release_year": release_year,
        "rating": _clean(row.get("rating")),
        "duration": _clean(row.get("duration")),
        "listed_in": listed,
        "listed_in_text": ", ".join(listed) if listed else None,
        "description": _clean(row.get("description")),
        "expanded_text": _clean(row.get("expanded_text")),
    }
    # drop empties
    doc = {k: v for k, v in doc.items() if v not in (None, "", [], {})}

    # stable ID
    _id = doc.get("show_id")
    if not _id:
        slug = _RE_WS.sub("", (doc.get("title") or "").lower())
        _id = f"{slug}_{release_year if release_year is not None else 'na'}_gen"
        doc["show_id"] = _id
    doc["_id"] = _id
    return doc

# =========================
# Embedding (E5 small v2)
# =========================
_embedder = None
_vector_dim = None

def get_embedder():
    global _embedder, _vector_dim
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
        v = _embedder.encode(["probe"], normalize_embeddings=True)
        _vector_dim = int(v.shape[1]) if hasattr(v, "shape") else len(v[0])
    return _embedder, _vector_dim

# =========================
# Bulk generator
# =========================
def generate_actions(df: pd.DataFrame, index: str) -> Iterator[Dict[str, Any]]:
    model, _ = get_embedder()
    # Normalize rows
    docs: List[Dict[str, Any]] = [row_normalize(r.to_dict()) for _, r in df.iterrows()]
    B = EMBED_BATCH
    # BGE-large uses no special prefixes, just clean text
    DOC_PREFIX = ""
    texts = [DOC_PREFIX + text_for_embedding(d) for d in docs]
    for i in range(0, len(docs), B):
        chunk_docs = docs[i:i+B]
        vecs = model.encode(texts[i:i+B], normalize_embeddings=True, batch_size=B, show_progress_bar=False)
        vecs_list = vecs.tolist() if hasattr(vecs, "tolist") else vecs
        for d, v in zip(chunk_docs, vecs_list):
            _id = d.pop("_id")
            yield {"_op_type": "index", "_index": index, "_id": _id, **d, "vector": v if isinstance(v, list) else list(v)}

# =========================
# Run
# =========================
def run():
    csv_path = os.getcwd() + f'/../../data/{CSV_NAME}'
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    client = make_client()
    _, dim = get_embedder()
    if dim is None:
        raise ValueError("Could not determine vector dimension from embedder.")
    recreate_index(client, INDEX_NAME, dim)

    ok, fail = bulk(client, generate_actions(df, INDEX_NAME), stats_only=True, raise_on_error=False)
    print(f"Bulk indexed OK={ok}, Fail={fail}")
    client.indices.refresh(index=INDEX_NAME)

if __name__ == "__main__":
    run()
