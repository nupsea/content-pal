#!/usr/bin/env python3
from __future__ import annotations
import os, re, math
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pandas as pd
from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import bulk

# --------------------------- CONFIG ---------------------------
OS_URL       = os.getenv("OS_URL", "https://localhost:9200")
OS_USER      = os.getenv("OS_USER")      
OS_PWD       = os.getenv("OS_PASS")       
VERIFY_TLS   = os.getenv("OS_VERIFY", 0) in (1, "1", "true", "True", "TRUE", True) 
INDEX_NAME   = os.getenv("OS_INDEX", "netflix_assets_v2")   # use a new name if you want a fresh index
CSV_NAME     = os.getenv("CSV_NAME", "netflix_titles_cleaned.csv")
EMBED_MODEL  = os.getenv("EMBED_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
EMBED_BATCH  = int(os.getenv("EMBED_BATCH", "256"))

# --------------------------- CLIENT ---------------------------
def make_client() -> OpenSearch:
    if OS_URL.startswith("https://"):
        return OpenSearch(
            OS_URL,
            http_auth=(OS_USER, OS_PWD) if OS_USER and OS_PWD else None,
            verify_certs=VERIFY_TLS,
            ssl_assert_hostname=VERIFY_TLS,
            ssl_show_warn=VERIFY_TLS,
            http_compress=True,
            connection_class=RequestsHttpConnection,
            timeout=60, max_retries=3, retry_on_timeout=True,
        )
    return OpenSearch(OS_URL, http_compress=True, timeout=60, max_retries=3, retry_on_timeout=True)

# --------------------------- MAPPING ---------------------------
def index_mapping(vector_dim: int) -> Dict[str, Any]:
    return {
        "settings": {
            "index": {"knn": True, "refresh_interval": "1s"}
        },
        "mappings": {
            "properties": {
                "show_id": {"type": "keyword"},
                "type": {"type": "keyword"},
                "type_text": {"type": "text"},

                "title": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
                "director": {"type": "text"},
                "cast": {"type": "text"},

                "cast_list": {"type": "keyword"},
                "country": {"type": "keyword"},
                "country_text": {"type": "text"},

                # store ISO date only here; keep raw in date_added_raw
                "date_added": {"type": "date"},
                "date_added_raw": {"type": "keyword"},

                "release_year": {"type": "integer"},
                "rating": {"type": "keyword"},
                "rating_text": {"type": "text"},

                "duration": {"type": "keyword"},
                "listed_in": {"type": "keyword"},
                "listed_in_text": {"type": "text"},

                "description": {"type": "text"},

                "vector": {
                    "type": "knn_vector",
                    "dimension": vector_dim,
                    "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"},
                },
            }
        }
    }

def _field_exists(client, index: str, field: str) -> bool:
    m = client.indices.get_mapping(index=index)
    props = m.get(index, {}).get("mappings", {}).get("properties", {}) or {}
    return field in props

def add_text_copies_mapping_safe(client, index: str) -> None:
    # If any of these exists, do nothing (avoid analyzer conflicts)
    fields = ("listed_in_text", "country_text", "rating_text", "type_text")
    if any(_field_exists(client, index, f) for f in fields):
        print("text-copy fields already present; skipping put_mapping")
        return
    body = {"properties": {
        "listed_in_text": {"type": "text"},
        "country_text":   {"type": "text"},
        "rating_text":    {"type": "text"},
        "type_text":      {"type": "text"},
    }}
    client.indices.put_mapping(index=index, body=body, ignore=400)


def ensure_index_legacy(client, index, vector_dim, create_new: bool = False) -> None:
    exists = client.indices.exists(index=index)
    if create_new or not (exists if isinstance(exists, bool) else exists):
        client.indices.create(index=index, body=index_mapping(vector_dim))
        print(f"created index {index}")
    else:
        print(f"using index {index}")
        add_text_copies_mapping_safe(client, index)  # see below
    client.indices.refresh(index=index)

def ensure_index_v3(client, index, vector_dim) -> None:
    # analyzer changes require a fresh index name or a delete/recreate
    if client.indices.exists(index=index):
        raise RuntimeError(
            f"Index {index} already exists. Delete it or use a new name when changing analyzers."
        )
    create_index_v3(client, index, vector_dim)
    client.indices.refresh(index=index)

def add_text_copies_mapping(client: OpenSearch, index: str) -> None:
    body = {"properties": {
        "listed_in_text": {"type": "text"},
        "country_text":   {"type": "text"},
        "rating_text":    {"type": "text"},
        "type_text":      {"type": "text"}
    }}
    try:
        client.indices.put_mapping(index=index, body=body)
    except Exception as e:
        print("put_mapping warning:", e)

# --------------------------- NORMALIZATION ---------------------------
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
    return None  # drop if unparseable (prevents mapper_parsing_exception)

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
        "type_text": _clean(row.get("type")),
        "title": _clean(row.get("title")),
        "director": _clean(row.get("director")),
        "cast": _clean(row.get("cast")),
        "cast_list": cast_list,
        "country": _clean(row.get("country")),
        "country_text": _clean(row.get("country")),
        "date_added_raw": raw_date,
        # only set date_added if parseable
        **({"date_added": iso_date} if iso_date else {}),
        "release_year": release_year,
        "rating": _clean(row.get("rating")),
        "rating_text": _clean(row.get("rating")),
        "duration": _clean(row.get("duration")),
        "listed_in": listed,
        "listed_in_text": ", ".join(listed) if listed else None,
        "description": _clean(row.get("description")),
    }
    # drop empties for cleanliness
    doc = {k: v for k, v in doc.items() if v not in (None, "", [], {})}

    # stable _id
    _id = doc.get("show_id")
    if not _id:
        slug = _RE_WS.sub("", (doc.get("title") or "").lower())
        _id = f"{slug}_{release_year if release_year is not None else 'na'}_gen"
        doc["show_id"] = _id
    doc["_id"] = _id
    return doc

# --------------------------- EMBEDDING ---------------------------
_embedder = None
_vector_dim = None

def get_embedder():
    global _embedder, _vector_dim
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)
        # probe dimension
        v = _embedder.encode(["probe"], normalize_embeddings=True)
        _vector_dim = int(v.shape[1]) if hasattr(v, "shape") else len(v[0])
    return _embedder, _vector_dim

# --------------------------- BULK INDEX ---------------------------
def generate_actions(df: pd.DataFrame, index: str) -> Iterator[Dict[str, Any]]:
    """Yields bulk index actions with vectors in batches."""
    model, dim = get_embedder()

    # normalize rows
    docs: List[Dict[str, Any]] = [row_normalize(r._asdict() if hasattr(r, "_asdict") else r.to_dict())
                                  for _, r in df.iterrows()]
    # batch encode texts
    texts = [text_for_embedding(d) for d in docs]
    for i in range(0, len(docs), EMBED_BATCH):
        chunk_docs = docs[i:i+EMBED_BATCH]
        chunk_texts = texts[i:i+EMBED_BATCH]
        vecs = model.encode(chunk_texts, normalize_embeddings=True, batch_size=EMBED_BATCH, show_progress_bar=False)
        # avoid per-row numpy overhead by single tolist
        vecs_list = vecs.tolist() if hasattr(vecs, "tolist") else vecs
        for d, v in zip(chunk_docs, vecs_list):
            _id = d.pop("_id")
            yield {
                "_op_type": "index",
                "_index": index,
                "_id": _id,
                **d,
                "vector": v if isinstance(v, list) else list(v),
            }


def create_index_v3(client, index: str, vector_dim: int) -> None:
    body = {
      "settings": {
        "index": {"knn": True, "refresh_interval": "1s"},
        "analysis": {
          "filter": {
            "edge_2_15": {"type": "edge_ngram", "min_gram": 2, "max_gram": 15},
            "shingle_2_3": {"type": "shingle", "min_shingle_size": 2, "max_shingle_size": 3, "output_unigrams": True},
            "genre_syns": {"type": "synonym_graph", "lenient": True, "synonyms": [
              "kids, children, family, preschool, toddler, cartoon, animated",
              "romcom, romantic comedy => romantic comedy",
              "sci-fi, scifi, science-fiction => science fiction",
              "doc, docs, documentary, documentaries => documentaries",
              "crime drama, crime series => crime",
              "thriller, suspense => thrillers",
              "kdrama, k-drama => korean tv shows",
              "anime, animation => anime",
              "friendships, friends, buddy => friendship",
              "teen, teens, teenagers, teenage, high school, high-school => teen",
              "piano, pianist, musician => pianist",
              "book club, bookclub => literature"
            ]},
            "country_syns": {"type": "synonym_graph", "lenient": True, "synonyms": [
              "uk, british, britain, england => united kingdom",
              "korean => south korea",
              "german => germany",
              "french => france",
              "spanish => spain",
              "italian => italy",
              "japanese => japan",
              "indian => india"
            ]},
            "rating_syns": {"type": "synonym_graph", "lenient": True, "synonyms": [
              "mature, adult => tv-ma",
              "nc17, nc-17 => nc-17"
            ]},
            "type_syns": {"type": "synonym_graph", "lenient": True, "synonyms": [
              "tv show, tvseries, series => tv show",
              "film => movie"
            ]}
          },
          "analyzer": {
            "title_prefix":  {"tokenizer": "standard", "filter": ["lowercase","edge_2_15"]},
            "title_search":  {"tokenizer": "standard", "filter": ["lowercase"]},
            "cast_prefix":   {"tokenizer": "standard", "filter": ["lowercase","edge_2_15"]},
            "cast_search":   {"tokenizer": "standard", "filter": ["lowercase"]},

            "english_desc":  {"tokenizer": "standard", "filter": ["lowercase","porter_stem","stop"]},
            "english_shing": {"tokenizer": "standard", "filter": ["lowercase","porter_stem","stop","shingle_2_3"]},

            "cat_anal":      {"tokenizer": "standard", "filter": ["lowercase","genre_syns"]},
            "country_anal":  {"tokenizer": "standard", "filter": ["lowercase","country_syns"]},
            "rating_anal":   {"tokenizer": "standard", "filter": ["lowercase","rating_syns"]},
            "type_anal":     {"tokenizer": "standard", "filter": ["lowercase","type_syns"]}
          }
        }
      },
      "mappings": {
        "properties": {
          "show_id": {"type": "keyword"},

          "type": {"type": "keyword"},
          "type_text": {"type": "text", "analyzer": "type_anal", "copy_to": ["search_blob"]},

          "title": {
            "type": "text",
            "analyzer": "title_search",
            "search_analyzer": "title_search",
            "copy_to": ["search_blob"],
            "fields": {
              "raw": {"type": "keyword"},
              "prefix": {"type": "text", "analyzer": "title_prefix", "search_analyzer": "title_search"}
            }
          },

          "director": {"type": "text", "copy_to": ["search_blob"]},

          "cast": {
            "type": "text",
            "analyzer": "cast_search",
            "search_analyzer": "cast_search",
            "copy_to": ["search_blob"],
            "fields": {
              "prefix": {"type": "text", "analyzer": "cast_prefix", "search_analyzer": "cast_search"}
            }
          },
          "cast_list": {"type": "keyword"},

          "country": {"type": "keyword"},
          "country_text": {"type": "text", "analyzer": "country_anal", "copy_to": ["search_blob"]},

          "date_added": {"type": "date"},
          "date_added_raw": {"type": "keyword"},

          "release_year": {"type": "integer"},
          "rating": {"type": "keyword"},
          "rating_text": {"type": "text", "analyzer": "rating_anal", "copy_to": ["search_blob"]},

          "duration": {"type": "keyword"},

          "listed_in": {"type": "keyword"},
          "listed_in_text": {"type": "text", "analyzer": "cat_anal", "copy_to": ["search_blob"]},

          "description": {
            "type": "text",
            "analyzer": "english_desc",
            "copy_to": ["search_blob"],
            "fields": {
              "shingles": {"type": "text", "analyzer": "english_shing"}
            }
          },

          "search_blob": {"type": "text", "analyzer": "english_shing"},

          "vector": {
            "type": "knn_vector",
            "dimension": vector_dim,
            "method": {"name": "hnsw", "engine": "faiss", "space_type": "cosinesimil"}
          }
        }
      }
    }
    client.indices.create(index=index, body=body)

def recreate_index_v3(client, index: str, vector_dim: int) -> None:
    try:
        if client.indices.exists(index=index):
            client.indices.delete(index=index, ignore=[400, 404])
            print(f"Deleted old index: {index}")
    except Exception as e:
        print(f"Warning while deleting {index}: {e}")
    create_index_v3(client, index, vector_dim)
    client.cluster.health(index=index, wait_for_status="yellow", request_timeout=90)
    client.indices.refresh(index=index)




def run():
    # load CSV
    print(" ** Loading Data.")
    csv_path = os.getcwd() + f'/../data/{CSV_NAME}'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    # prep client and index
    client = make_client()
    _, dim = get_embedder()
    if dim is None:
        raise ValueError("Could not determine vector dimension from embedder.")

    print("Deleting old index and re-creating..")
    recreate_index_v3(client, INDEX_NAME, dim)
    # ensure_index_v3(client, INDEX_NAME, dim)

    # bulk index (re-upsert)
    ok, fail = bulk(client, generate_actions(df, INDEX_NAME), stats_only=True, raise_on_error=False)
    print(f"Bulk indexed OK={ok}, Fail={fail}")
    client.indices.refresh(index=INDEX_NAME)

if __name__ == "__main__":
    run()
