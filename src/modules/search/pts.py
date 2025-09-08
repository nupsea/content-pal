# modules/search/pts.py
import re
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
from .core import ContentSearchSystem, SearchResult, SearchConfig

class PretrainedSemanticSearch:
    def __init__(
        self,
        backend_type: str = "minsearch",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        per_head_k: int = 200,
        max_candidates_cap: int = 1000,
        alpha_semantic: float = 0.9,
        rrf_k: int = 60,
        doc_char_budget: int = 1200,
    ):
        self.base = ContentSearchSystem(backend_type=backend_type)
        self.per_head_k = per_head_k
        self.max_candidates_cap = max_candidates_cap
        self.alpha_semantic = float(alpha_semantic)
        self.rrf_k = rrf_k
        self.doc_char_budget = doc_char_budget
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._show_id_lc: set[str] = set()
        self._title_to_show_id: Dict[str, str] = {}
        self.indexed = False
        try:
            self.ce = CrossEncoder(model_name)
        except Exception:
            self.ce = None

    # ---------- indexing ----------
    def index_data(self, csv_path: str):
        self.base.index_data(csv_path)
        df = pd.read_csv(csv_path, encoding="latin-1").fillna("")
        for _, row in df.iterrows():
            sid = str(row["show_id"]).strip()
            title = str(row.get("title", "")).strip()
            self.documents[sid] = {"text": self._doc_text(row), "title": title, "meta": row.to_dict()}
            self._show_id_lc.add(sid.lower())
            if title:
                self._title_to_show_id[title.lower()] = sid
        self.indexed = True

    def _doc_text(self, row: pd.Series) -> str:
        parts = []
        title = str(row.get("title", "")).strip()
        ty = " ".join([str(row.get("type","")).strip(), str(row.get("release_year","")).strip()]).strip()
        cast = str(row.get("cast", "")).strip()
        genres = str(row.get("listed_in", "")).strip()
        desc = str(row.get("description","")).strip()
        if title:  parts.append(f"Title: {title}")
        if ty:     parts.append(f"Type/Year: {ty}")
        if cast:   parts.append(f"Cast: {cast}")
        if genres: parts.append(f"Genres: {genres}")
        if desc:   parts.append(f"Plot: {desc}")
        txt = " | ".join(parts)
        return (txt[:self.doc_char_budget].rsplit(" ",1)[0]+" ...") if len(txt) > self.doc_char_budget else txt

    # ---------- query prep (lean) ----------
    SYNONYMS = {
        "sci fi": ["science fiction", "scifi", "sci-fi"],
        "romcom": ["rom com", "romance comedy"],
        "feel good": ["uplifting", "heartwarming", "wholesome", "happy ending"],
        "kids": ["children & family movies", "kids tv", "children and family movies"],
        "teen": ["teen tv shows", "teen"],         
        "international": ["international movies", "international tv shows"],
        "mysteries": ["mystery", "mysteries", "suspense", "detective"],     
    }
    DECADES = {"20s":"1920s","30s":"1930s","40s":"1940s","50s":"1950s","60s":"1960s",
               "70s":"1970s","80s":"1980s","90s":"1990s","2000s":"2000s","2010s":"2010s","2020s":"2020s"}
    
    RATINGS = {"G","PG","PG-13","R","NC-17","TV-Y","TV-Y7","TV-G","TV-PG","TV-14","TV-MA"}

    def _extract_years(self, qn: str) -> list[str]:
        return re.findall(r"\b(?:19|20)\d{2}\b", qn)  # e.g., 2017

    def _extract_ratings(self, raw_q: str) -> list[str]:
        q = raw_q.upper()
        found = []
        for r in self.RATINGS:
            if r in q or r.replace("-", " ") in q:
                found.append(r)                     # e.g., "TV-MA"
                if "-" in r: found.append(r.replace("-", " "))      # e.g., "TV 14"
        return list(dict.fromkeys(found))


    def _normalize(self, q: str) -> str:
        q = q.lower()
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _expand(self, q: str) -> str:
        qn = self._normalize(q)
        bag = set(qn.split())
        extra = []
        for k, exps in self.SYNONYMS.items():
            if all(t in bag for t in k.split()):
                extra += exps
        for k, v in self.DECADES.items():
            if k in bag:
                extra.append(v)

        extra.extend(self._extract_ratings(q))   # ratings
        extra.extend(self._extract_years(qn))    # years like 2017

        return (qn + " " + " ".join(extra)).strip() if extra else qn

    # ---------- ID repair ----------
    def _repair_id(self, sr: SearchResult) -> SearchResult:
        """
        Ensure sr.id is a catalog show_id:
        - if already a show_id (case-insensitive), keep
        - else try metadata['show_id'] / ['_id'] / ['id']
        - else try title-based map
        """
        cid = str(getattr(sr, "id", "")).strip()
        if cid and cid.lower() in self._show_id_lc:
            return sr  # already good

        meta = None
        # try common payload attributes
        for attr in ("payload", "meta", "metadata", "document"):
            if hasattr(sr, attr):
                val = getattr(sr, attr)
                if isinstance(val, dict):
                    meta = val
                    break

        # try explicit id fields from payload
        if isinstance(meta, dict):
            for key in ("show_id", "_id", "id"):
                if key in meta:
                    sid = str(meta[key]).strip()
                    if sid and sid.lower() in self._show_id_lc:
                        sr.id = sid
                        return sr
            # last resort: title map
            title = meta.get("title") or getattr(sr, "title", None)
            if title:
                sid = self._title_to_show_id.get(str(title).strip().lower())
                if sid:
                    sr.id = sid
                    return sr

        # nothing worked; leave as-is
        return sr

    # ---------- multi-head retrieval ----------
    def _heads(self) -> List[Tuple[str, SearchConfig]]:
        K = self.per_head_k
        return [
            ("people", SearchConfig(boost_weights={"title":2.0,"cast":5.0,"director":1.5,"description":1.0,"listed_in":1.0}, max_results=K)),
            ("plot",   SearchConfig(boost_weights={"title":2.0,"description":5.0,"cast":1.0,"listed_in":1.5}, max_results=K)),
            ("genre",  SearchConfig(boost_weights={"listed_in":4.0,"title":2.0,"description":1.5}, max_results=K)),
            ("title",  SearchConfig(boost_weights={"title":5.0,"description":1.0}, max_results=K)),
            ("meta",   SearchConfig(boost_weights={"rating":6.0, "release_year":3.0, "listed_in":2.5, "country":2.0, "title":1.0}, max_results=K)),
        ]

    def _rrf_fuse(self, per_head: Dict[str, List[SearchResult]]) -> List[SearchResult]:
        k = self.rrf_k
        scores: Dict[str, float] = {}
        best: Dict[str, SearchResult] = {}
        for _, results in per_head.items():
            for r, s in enumerate(results, 1):
                s = self._repair_id(s)  # << ID fix here
                sid = s.id
                scores[sid] = scores.get(sid, 0.0) + 1.0/(k+r)
                if sid not in best or getattr(s, "score", 0.0) > getattr(best[sid], "score", 0.0):
                    best[sid] = s
        fused = list(best.values())
        fused.sort(key=lambda sr: (-scores.get(sr.id, 0.0),
                                   -int(self.documents.get(sr.id,{}).get("meta",{}).get("release_year", -1) or -1),
                                   str(self.documents.get(sr.id,{}).get("title","")).lower()))
        return fused

    def _candidates(self, query: str) -> List[SearchResult]:
        qx = self._expand(query)
        per_head = {name: (self.base.search(qx, cfg) or []) for name, cfg in self._heads()}
        fused = self._rrf_fuse(per_head)
        return fused[: self.max_candidates_cap]

    # ---------- CE rerank ----------
    @staticmethod
    def _minmax(xs: List[float]) -> List[float]:
        if not xs: return []
        lo, hi = float(min(xs)), float(max(xs))
        if hi <= lo: return [0.5]*len(xs)
        return [(x - lo)/(hi - lo) for x in xs]

    def _ce_rerank(self, query: str, cands: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not self.ce or not cands:
            return cands[:top_k]
        pairs, keep = [], []
        for c in cands:
            d = self.documents.get(c.id)
            if not d:  # if ID still unmatched, skip; should be rare after _repair_id
                continue
            pairs.append([query, d["text"]])
            keep.append(c)
        if not keep:
            return cands[:top_k]
        scores = [float(s) for s in self.ce.predict(pairs, batch_size=64, show_progress_bar=False)]
        ce_n = self._minmax(scores)
        base_n = self._minmax([float(getattr(c,"score",0.0)) for c in keep])
        a = self.alpha_semantic
        for i,c in enumerate(keep):
            c.score = a*ce_n[i] + (1-a)*base_n[i]
        def key(c: SearchResult):
            m = self.documents.get(c.id,{}).get("meta",{})
            yr = int(m.get("release_year", -1)) if str(m.get("release_year","")).isdigit() else -1
            ttl = str(self.documents.get(c.id,{}).get("title","")).lower()
            return (-c.score, -yr, ttl)
        keep.sort(key=key)
        return keep[:top_k]

    # ---------- public ----------
    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        if not self.indexed: raise RuntimeError("index_data() first")
        cands = self._candidates(query)
        return self._ce_rerank(query, cands, top_k)
