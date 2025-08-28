from datetime import datetime
import re

_DATE_PATTERNS = [
    "%Y-%m-%d",        # 2021-09-25
    "%B %d, %Y",       # September 25, 2021
    "%b %d, %Y",       # Sep 25, 2021
    "%B %d %Y",        # September 25 2021 (no comma)
    "%b %d %Y",        # Sep 25 2021 (no comma)
]

def to_iso_date(s: str | None) -> str | None:
    if not s:
        return None
    s2 = str(s).strip()
    for fmt in _DATE_PATTERNS:
        try:
            return datetime.strptime(s2, fmt).date().isoformat()
        except Exception:
            pass
    return None

def add_text_copies_mapping(client, index: str) -> None:
    body = {"properties": {
        "listed_in_text": {"type": "text"},
        "country_text":   {"type": "text"},
        "rating_text":    {"type": "text"},
        "type_text":      {"type": "text"}
    }}
    client.indices.put_mapping(index=index, body=body, ignore=400)


def _clean(s):
    if s is None: return None
    s2 = str(s).strip()
    return s2 or None

def _split_csv(s):
    s = _clean(s)
    return [p.strip() for p in s.split(",")] if s else []

def row_normalize(row: dict) -> dict:
    listed = _split_csv(row.get("listed_in"))
    cast_list = _split_csv(row.get("cast"))
    ry = _clean(row.get("release_year"))
    release_year = int(ry) if ry and ry.isdigit() else None
    
    raw_date = _clean(row.get("date_added"))
    iso_date = to_iso_date(raw_date)

    doc = {
        "show_id": _clean(row.get("show_id")),
        "type": _clean(row.get("type")),
        "type_text": _clean(row.get("type")),
        "title": _clean(row.get("title")),
        "director": _clean(row.get("director")),
        "cast": _clean(row.get("cast")),
        "cast_list": cast_list,
        "country": _clean(row.get("country")),
        "country_text": _clean(row.get("country")),
        "release_year": release_year,
        "rating": _clean(row.get("rating")),
        "rating_text": _clean(row.get("rating")),
        "duration": _clean(row.get("duration")),
        "listed_in": listed,
        "listed_in_text": ", ".join(listed) if listed else None,
        "description": _clean(row.get("description")),
        "date_added_raw": raw_date,     
        # only set date_added if we could parse it
        **({"date_added": iso_date} if iso_date else {}),
    }
    # drop empties
    doc = {k: v for k, v in doc.items() if v not in (None, "", [], {})}
    doc["_id"] = doc.get("show_id") or (re.sub(r"\s+", "", (doc.get("title") or "").lower()) + "_gen")
    return doc


def text_for_embedding(d: dict) -> str:
    parts = []
    for k in ("title","description","director","type","country","rating"):
        v = d.get(k); parts.append(v) if v else None
    if d.get("cast_list"): parts.append(", ".join(d["cast_list"]))
    if d.get("listed_in"): parts.append(", ".join(d["listed_in"]))
    return " | ".join(parts)
