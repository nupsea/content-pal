# modules/search/schema_aware_semantic.py
# Data-driven hybrid retriever:
#   schema-aware expansion (from CSV) + PRF → multi-head BM25 → RRF → CE rerank.
# No hard-coded domain synonyms; expansions come from the catalog values.

from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import re, difflib
import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder
# Optional: dense head (off by default). Only used if you set use_dense_head=True
# from sentence_transformers import SentenceTransformer
# import faiss

from .core import ContentSearchSystem, SearchResult, SearchConfig


class SchemaAwareSemanticSearch:
    def __init__(
        self,
        backend_type: str = "minsearch",
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        per_head_k: int = 250,
        max_candidates_cap: int = 1500,
        rrf_k: int = 60,
        alpha_semantic: float = 0.95,
        doc_char_budget: int = 1200,
        # PRF
        prf_m: int = 40,              # take top-M from the strongest head for expansion
        prf_top_facets: int = 3,      # how many facets per type to add
        # Dense head (optional, default OFF)
        use_dense_head: bool = False,
        dense_k: int = 100,
    ):
        self.base = ContentSearchSystem(backend_type=backend_type)
        self.per_head_k = per_head_k
        self.max_candidates_cap = max_candidates_cap
        self.rrf_k = rrf_k
        self.alpha = float(alpha_semantic)
        self.doc_char_budget = int(doc_char_budget)
        self.prf_m = prf_m
        self.prf_top_facets = prf_top_facets

        self.docs: Dict[str, Dict[str, Any]] = {}
        self._show_id_lc: set[str] = set()

        # catalog-derived vocab (no hard-coded lists)
        self.vocab = {
            "ratings": set(),
            "types": set(),
            "genres": set(),
            "countries": set(),
            "people": set(),  # cast + directors, lowercased
        }

        # CE
        try:
            self.ce = CrossEncoder(model_name)
        except Exception:
            self.ce = None

        # Dense head (off by default)
        self.use_dense_head = use_dense_head
        self.dense_k = dense_k
        self._dense = None  # (model, index, id_list), built on demand

        self.indexed = False

    # ---------------- indexing ----------------
    def index_data(self, csv_path: str):
        self.base.index_data(csv_path)
        df = pd.read_csv(csv_path, encoding="latin-1").fillna("")

        # build document texts + vocab from data
        for _, row in df.iterrows():
            sid = str(row["show_id"]).strip()
            title = str(row.get("title", "")).strip()
            cast = [c.strip() for c in str(row.get("cast", "")).split(",") if c.strip()]
            directors = [d.strip() for d in str(row.get("director", "")).split(",") if d.strip()]
            countries = [c.strip() for c in str(row.get("country", "")).split(",") if c.strip()]
            genres = [g.strip() for g in str(row.get("listed_in", "")).split(",") if g.strip()]
            rating = str(row.get("rating", "")).strip()
            ryear = str(row.get("release_year", "")).strip()

            self.docs[sid] = {
                "text": self._doc_text(row),
                "title": title,
                "meta": row.to_dict(),
                "facets": {
                    "people": [p for p in cast + directors if p],
                    "genres": genres,
                    "countries": countries,
                    "rating": rating,
                    "year": ryear,
                },
            }
            self._show_id_lc.add(sid.lower())

            if rating: self.vocab["ratings"].add(rating.upper())
            if row.get("type"): self.vocab["types"].add(str(row["type"]).strip().lower())
            for g in genres: self.vocab["genres"].add(g.lower())
            for c in countries: self.vocab["countries"].add(c.lower())
            for n in cast + directors:
                if n: self.vocab["people"].add(n.lower())

        # Optional dense head build (kept OFF unless use_dense_head=True is passed)
        if self.use_dense_head:
            try:
                from sentence_transformers import SentenceTransformer
                import faiss
                model = SentenceTransformer("intfloat/e5-small-v2")
                ids, embs = [], []
                for sid, d in self.docs.items():
                    ids.append(sid)
                    embs.append(model.encode(d["text"], normalize_embeddings=True))
                embs = np.asarray(embs, dtype="float32")
                idx = faiss.IndexFlatIP(embs.shape[1]); idx.add(embs)
                self._dense = (model, idx, ids)
            except Exception:
                self.use_dense_head = False

        self.indexed = True

    def _doc_text(self, row: pd.Series) -> str:
        parts = []
        t = str(row.get("title","")).strip()
        ty = " ".join([str(row.get("type","")).strip(), str(row.get("release_year","")).strip()]).strip()
        cast = str(row.get("cast","")).strip()
        genres = str(row.get("listed_in","")).strip()
        desc = str(row.get("description","")).strip()
        if t: parts.append(f"Title: {t}")
        if ty: parts.append(f"Type/Year: {ty}")
        if cast: parts.append(f"Cast: {cast}")
        if genres: parts.append(f"Genres: {genres}")
        if desc: parts.append(f"Plot: {desc}")
        txt = " | ".join(parts)
        return (txt[:self.doc_char_budget].rsplit(" ",1)[0] + " ...") if len(txt) > self.doc_char_budget else txt

    # ---------------- query prep (schema-driven, no hard-coded synonyms) ----------------
    @staticmethod
    def _normalize(q: str) -> str:
        q = q.lower()
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()
        return q

    def _extract_years(self, q_norm: str) -> List[str]:
        return re.findall(r"\b(?:19|20)\d{2}\b", q_norm)

    def _extract_decades(self, q_norm: str) -> List[str]:
        dec = []
        for m in re.finditer(r"\b(19|20)\d0s\b", q_norm):
            dec.append(m.group(0))
        # compact forms like "80s"
        for m in re.finditer(r"\b([5-9]0s)\b", q_norm):
            dec.append(m.group(0))  # "80s" etc.; your corpus may have those tokens in text
        return list(dict.fromkeys(dec))

    def _match_catalog_values(self, q_tokens: List[str], candidates: List[str], topn=5, cutoff=0.85) -> List[str]:
        """Fuzzy match query tokens to catalog values (genres/ratings/countries/people).
           Uses difflib (Levenshtein-free) to avoid hard-coding."""
        matched = []
        joined = [" ".join(q_tokens[i:i+2]) for i in range(len(q_tokens)-1)] + q_tokens
        # unique, longer phrases first
        for tok in sorted(set(joined), key=len, reverse=True):
            # difflib works on lowercase strings
            # candidates must already be lowercased
            close = difflib.get_close_matches(tok, candidates, n=topn, cutoff=cutoff)
            matched.extend(close)
        return list(dict.fromkeys(matched))

    def _schema_expand(self, q: str) -> Tuple[str, Dict[str, List[str]]]:
        """Return expanded query string and parsed signals (for logging / future features)."""
        qn = self._normalize(q)
        tokens = qn.split()

        years = self._extract_years(qn)
        decades = self._extract_decades(qn)

        # Candidate sets from catalog (lowercased)
        genres_cand = list(self.vocab["genres"])
        ratings_cand = [r.lower() for r in self.vocab["ratings"]]
        countries_cand = list(self.vocab["countries"])
        people_cand = list(self.vocab["people"])

        genres_hit = self._match_catalog_values(tokens, genres_cand, topn=3, cutoff=0.82)
        ratings_hit = self._match_catalog_values(tokens, ratings_cand, topn=3, cutoff=0.90)
        countries_hit = self._match_catalog_values(tokens, countries_cand, topn=3, cutoff=0.88)
        # for people, require higher cutoff to avoid spurious matches
        people_hit = self._match_catalog_values(tokens, people_cand, topn=3, cutoff=0.93)

        # We only append *catalog* tokens (no arbitrary synonyms)
        extra = []
        extra += years + decades
        extra += genres_hit + ratings_hit + countries_hit
        # People tokens are useful, but some analyzers split names; keep them as-is
        extra += people_hit

        expanded = (qn + " " + " ".join(extra)).strip() if extra else qn
        signals = {
            "years": years, "decades": decades,
            "genres": genres_hit, "ratings": ratings_hit,
            "countries": countries_hit, "people": people_hit
        }
        return expanded, signals

    # ---------------- PRF: mine facets from top lexical hits ----------------
    def _prf_expand(self, qx: str, strong_head_results: List[SearchResult]) -> str:
        """Rocchio-style expansion from top-M docs: add the most common facets (people/genres/rating)."""
        M = min(self.prf_m, len(strong_head_results))
        if M <= 0: return qx
        ppl = Counter(); gen = Counter(); rat = Counter(); yrs = Counter(); ctry = Counter()
        for sr in strong_head_results[:M]:
            d = self.docs.get(str(sr.id))
            if not d: continue
            f = d["facets"]
            for p in f.get("people", []): ppl[p.lower()] += 1
            for g in f.get("genres", []): gen[g.lower()] += 1
            if f.get("rating"): rat[f["rating"].lower()] += 1
            if f.get("year"): yrs[str(f["year"]).lower()] += 1
            for c in f.get("countries", []): ctry[c.lower()] += 1
        # take top-k per facet type
        k = self.prf_top_facets
        extra = []
        extra += [w for w,_ in ppl.most_common(k)]
        extra += [w for w,_ in gen.most_common(k)]
        extra += [w for w,_ in rat.most_common(1)]
        extra += [w for w,_ in yrs.most_common(1)]
        extra += [w for w,_ in ctry.most_common(k)]
        extra = [x for x in extra if x]  # dedup while preserving order
        dedup = []
        seen = set()
        for x in extra:
            if x not in seen:
                dedup.append(x); seen.add(x)
        return (qx + " " + " ".join(dedup)).strip() if dedup else qx

    # ---------- entity-field detection ----------
    def _extract_entities(self, q: str) -> Dict[str, List[str]]:
        """Extract entities and their target fields from query"""
        q_lower = q.lower()
        entities = {
            'directors': [],
            'actors': [],
            'genres': [],
            'years': []
        }
        
        # Director patterns - more comprehensive
        director_patterns = [
            r"directed by ([^,\.]+)",
            r"director ([^,\.]+)", 
            r"films? by ([^,\.]+)",
            r"movies? by ([^,\.]+)"
        ]
        
        # Special patterns for "Name films/movies" - handle both proper case and lowercase
        name_patterns = [
            r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+films?$",     # "Christopher Nolan films"
            r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+movies?$"     # "Christopher Nolan movies" 
        ]
        
        # Case-insensitive patterns for lowercase input
        name_patterns_lower = [
            r"\b([a-z]+ [a-z]+(?:\s+[a-z]+)*)\s+films?$",     # "christopher nolan films"
            r"\b([a-z]+ [a-z]+(?:\s+[a-z]+)*)\s+movies?$"     # "christopher nolan movies" 
        ]
        
        for pattern in director_patterns:
            matches = re.findall(pattern, q_lower)
            entities['directors'].extend([m.strip() for m in matches if m.strip()])
        
        # Check name patterns on original query (with proper capitalization)
        for pattern in name_patterns:
            matches = re.findall(pattern, q)
            entities['directors'].extend([m.strip() for m in matches if m.strip()])
        
        # Check lowercase name patterns and title case the results
        for pattern in name_patterns_lower:
            matches = re.findall(pattern, q_lower)
            entities['directors'].extend([m.strip().title() for m in matches if m.strip()])
        
        # Actor patterns (only if not already detected as director)
        if not entities['directors']:  # Only check for actors if no directors found
            actor_patterns = [
                r"starring ([^,\.]+)",
                r"with ([^,\.]+)", 
                r"featuring ([^,\.]+)"
            ]
            
            # Special patterns for "Name movies/films" - handle both proper case and lowercase
            actor_name_patterns = [
                r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+movies?$",  # "Leonardo DiCaprio movies"
                r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+films?$"    # "Tom Hanks films"
            ]
            
            # Case-insensitive patterns for lowercase input
            actor_name_patterns_lower = [
                r"\b([a-z]+ [a-z]+(?:\s+[a-z]+)*)\s+movies?$",  # "leonardo dicaprio movies"
                r"\b([a-z]+ [a-z]+(?:\s+[a-z]+)*)\s+films?$"    # "tom hanks films"
            ]
            
            for pattern in actor_patterns:
                matches = re.findall(pattern, q_lower)
                for match in matches:
                    name = match.strip()
                    # Only add if it looks like a person name (2+ words, reasonable length)
                    if len(name.split()) >= 2 and 3 <= len(name) <= 30:
                        entities['actors'].append(name)
            
            # Check actor name patterns on original query (with proper capitalization)
            for pattern in actor_name_patterns:
                matches = re.findall(pattern, q)
                entities['actors'].extend([m.strip() for m in matches if m.strip()])
            
            # Check lowercase actor name patterns and title case the results
            for pattern in actor_name_patterns_lower:
                matches = re.findall(pattern, q_lower)
                entities['actors'].extend([m.strip().title() for m in matches if m.strip()])
        
        # Genre detection (using existing vocab)
        for genre in self.vocab.get("genres", []):
            if genre in q_lower:
                entities['genres'].append(genre)
        
        # Year detection - more comprehensive
        year_patterns = [
            r'\b(19\d{2})\b',  # 1900s years
            r'\b(20\d{2})\b',  # 2000s years
            r'\b(19\d{2})-(20\d{2})\b',  # Year ranges like 1999-2005
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, q)
            for match in matches:
                if isinstance(match, tuple):
                    entities['years'].extend([y for y in match if y])
                else:
                    entities['years'].append(match)
        
        # Temporal constraints detection
        temporal_constraints = {}
        if 'before' in q_lower or 'prior to' in q_lower:
            # Extract year after "before" or "prior to"
            before_patterns = [
                r'(?:before|prior to)\s+(\d{4})',
                r'(?:before|prior to)\s+the year\s+(\d{4})'
            ]
            for pattern in before_patterns:
                matches = re.findall(pattern, q_lower)
                if matches:
                    temporal_constraints['before'] = int(matches[0])
                    break
        
        if 'after' in q_lower or 'since' in q_lower:
            after_patterns = [
                r'(?:after|since)\s+(\d{4})',
                r'(?:after|since)\s+the year\s+(\d{4})'
            ]
            for pattern in after_patterns:
                matches = re.findall(pattern, q_lower)
                if matches:
                    temporal_constraints['after'] = int(matches[0])
                    break
        
        entities['temporal_constraints'] = temporal_constraints
        
        # Clean duplicates (but preserve temporal_constraints dict)
        for key in entities:
            if key != 'temporal_constraints' and isinstance(entities[key], list):
                entities[key] = list(dict.fromkeys(entities[key]))
            
        return entities
    
    def _create_entity_searches(self, entities: Dict[str, List[str]], base_query: str) -> List[Tuple[str, str, SearchConfig]]:
        """Create targeted searches for detected entities"""
        entity_searches = []
        
        # Director-specific searches
        for director in entities['directors']:
            query = f'director:"{director}" OR {director}'
            config = SearchConfig(
                boost_weights={"director": 20.0, "title": 2.0, "cast": 1.0, "description": 1.0},
                max_results=50
            )
            entity_searches.append((f"director_{director}", query, config))
        
        # Actor-specific searches  
        for actor in entities['actors']:
            query = f'cast:"{actor}" OR {actor}'
            config = SearchConfig(
                boost_weights={"cast": 20.0, "title": 2.0, "director": 1.0, "description": 1.0}, 
                max_results=50
            )
            entity_searches.append((f"actor_{actor}", query, config))
            
        # Year/temporal constraint searches
        temporal_constraints = entities.get('temporal_constraints', {})
        if temporal_constraints:
            query_parts = []
            
            if 'before' in temporal_constraints:
                year = temporal_constraints['before']
                query_parts.append(f"release_year:[* TO {year-1}]")
            
            if 'after' in temporal_constraints:
                year = temporal_constraints['after']  
                query_parts.append(f"release_year:[{year+1} TO *]")
                
            if query_parts:
                temporal_query = " AND ".join(query_parts)
                config = SearchConfig(
                    boost_weights={"release_year": 10.0, "title": 2.0, "description": 1.0},
                    max_results=50
                )
                entity_searches.append(("temporal_constraint", temporal_query, config))
        
        # Explicit year searches
        for year in entities['years']:
            if year.isdigit() and len(year) == 4:
                query = f'release_year:{year}'
                config = SearchConfig(
                    boost_weights={"release_year": 15.0, "title": 2.0, "description": 1.0},
                    max_results=50
                )
                entity_searches.append((f"year_{year}", query, config))

        # Genre-specific searches
        for genre in entities['genres']:
            query = f'listed_in:"{genre}" OR {genre}'
            config = SearchConfig(
                boost_weights={"listed_in": 15.0, "title": 3.0, "description": 2.0},
                max_results=50
            )
            entity_searches.append((f"genre_{genre}", query, config))
        
        return entity_searches

    # ---------------- multi-head retrieval ----------------
    def _heads(self) -> List[Tuple[str, SearchConfig]]:
        K = self.per_head_k
        return [
            ("plot",   SearchConfig(boost_weights={"title":2.0,"description":5.0,"listed_in":1.5}, max_results=K)),
            ("people", SearchConfig(boost_weights={"title":2.0,"cast":5.0,"director":3.0,"description":1.0}, max_results=K)),
            ("facet",  SearchConfig(boost_weights={"listed_in":4.0,"rating":6.0,"release_year":3.0,"country":2.5,"title":1.0}, max_results=K)),
            ("title",  SearchConfig(boost_weights={"title":5.0,"description":1.0}, max_results=K)),
        ]

    def _rrf(self, rank_lists: List[List[SearchResult]]) -> List[SearchResult]:
        score = defaultdict(float); pick: Dict[str, SearchResult] = {}
        k = self.rrf_k
        for lst in rank_lists:
            for r, s in enumerate(lst, 1):
                sid = str(s.id)
                if not sid or sid.lower() not in self._show_id_lc:  # require canonical id
                    continue
                score[sid] += 1.0 / (k + r)
                if sid not in pick or getattr(s, "score", 0.0) > getattr(pick[sid], "score", 0.0):
                    pick[sid] = SearchResult(id=sid, title=getattr(s, "title", ""), score=getattr(s, "score", 0.0), content_type=getattr(s, "content_type", ""), metadata=getattr(s, "metadata", {}))
        fused = list(pick.values())
        fused.sort(key=lambda s: (-score.get(s.id, 0.0),
                                  -int(self.docs.get(s.id,{}).get("meta",{}).get("release_year",-1) or -1),
                                  str(self.docs.get(s.id,{}).get("title","")).lower()))
        return fused

    def _dense_candidates(self, qx: str) -> List[str]:
        if not (self.use_dense_head and self._dense): return []
        model, idx, ids = self._dense
        qvec = model.encode(qx, normalize_embeddings=True).astype("float32")
        import numpy as np
        D, I = idx.search(np.asarray([qvec]), self.dense_k)
        return [ids[i] for i in I[0] if i >= 0]

    def _candidates(self, q: str) -> List[SearchResult]:
        # schema-driven expansion
        qx, signals = self._schema_expand(q)
        
        # entity detection for field-specific search
        entities = self._extract_entities(q)
        has_entities = any(entities.values())

        # primary heads (standard semantic search)
        lists = []
        head_cfgs = self._heads()
        # run plot head first to feed PRF
        plot_res = self.base.search(qx, head_cfgs[0][1]) or []
        # PRF expansion (data-driven, from top plot docs)
        qx_prf = self._prf_expand(qx, plot_res)

        # get the rest using PRF-augmented query
        lists.append(plot_res)
        for name, cfg in head_cfgs[1:]:
            res = self.base.search(qx_prf, cfg) or []
            lists.append(res)

        # entity-specific searches (if entities detected)
        entity_results = []
        entity_search_count = 0
        if has_entities:
            entity_searches = self._create_entity_searches(entities, qx_prf)
            for search_name, entity_query, entity_config in entity_searches:
                try:
                    entity_res = self.base.search(entity_query, entity_config) or []
                    if entity_res:  # Only add if we got results
                        entity_results.extend(entity_res)
                        lists.append(entity_res)
                        entity_search_count += 1
                except:
                    # Fallback to simple search if field queries fail
                    simple_query = entity_query.split(' OR ')[1] if ' OR ' in entity_query else entity_query
                    entity_res = self.base.search(simple_query, entity_config) or []
                    if entity_res:
                        entity_results.extend(entity_res)
                        lists.append(entity_res)
                        entity_search_count += 1

        # Prioritize entity results when entities are detected
        if has_entities and entity_results:
            # Deduplicate entity results and sort by score
            entity_by_id = {}
            for result in entity_results:
                if result.id not in entity_by_id or result.score > entity_by_id[result.id].score:
                    entity_by_id[result.id] = result
            
            entity_priority = sorted(entity_by_id.values(), key=lambda x: -x.score)[:20]
            
            # Get regular search results (excluding entity searches from lists)
            regular_lists = lists[:-entity_search_count] if entity_search_count > 0 else lists
            regular_fused = self._rrf(regular_lists)
            
            # Remove entity results from regular results to avoid duplicates
            entity_ids = {r.id for r in entity_priority}
            regular_remaining = [r for r in regular_fused if r.id not in entity_ids]
            
            # Combine: entity results first, then regular results
            fused = entity_priority + regular_remaining
        else:
            fused = self._rrf(lists)

        # optional dense union
        if self.use_dense_head:
            dense_ids = set(self._dense_candidates(qx_prf))
            have = {c.id for c in fused}
            for sid in dense_ids - have:
                fused.append(SearchResult(id=sid, title="", score=0.1, content_type="", metadata={}))

        # cap
        fused.sort(key=lambda s: -getattr(s, "score", 0.0))
        return fused[: self.max_candidates_cap]

    # ---------------- CE rerank ----------------
    @staticmethod
    def _minmax(xs: List[float]) -> List[float]:
        if not xs: return []
        lo, hi = float(min(xs)), float(max(xs))
        if hi <= lo: return [0.5] * len(xs)
        return [(x - lo) / (hi - lo) for x in xs]

    def _ce_rerank(self, query: str, cands: List[SearchResult], top_k: int) -> List[SearchResult]:
        if not self.ce or not cands:
            return cands[:top_k]
        
        # Check if this is an entity query that should preserve entity matches
        entities = self._extract_entities(query)
        has_entity_constraints = any(entities.values())
        
        pairs, keep = [], []
        for c in cands:
            d = self.docs.get(c.id)
            if not d: continue
            pairs.append([query, d["text"]])
            keep.append(c)
        
        scores = [float(s) for s in self.ce.predict(pairs, batch_size=64, show_progress_bar=False)]
        ce_n = self._minmax(scores)
        base_n = self._minmax([float(getattr(c, "score", 0.0)) for c in keep])
        a = self.alpha
        
        # Separate entity matches from non-entity matches
        entity_matches = []
        non_entity_matches = []
        
        for i, c in enumerate(keep):
            # Calculate base CE score
            base_score = a * ce_n[i] + (1 - a) * base_n[i]
            c.score = base_score
            
            # Check if this result matches detected entities
            is_entity_match = False
            if has_entity_constraints:
                metadata = c.metadata
                
                # Check for director matches
                for director in entities.get('directors', []):
                    if director.lower() in metadata.get('director', '').lower():
                        is_entity_match = True
                        break
                
                # Check for actor matches
                if not is_entity_match:
                    for actor in entities.get('actors', []):
                        if actor.lower() in metadata.get('cast', '').lower():
                            is_entity_match = True
                            break
                
                # Check for temporal constraint matches
                if not is_entity_match:
                    temporal_constraints = entities.get('temporal_constraints', {})
                    if temporal_constraints and 'release_year' in metadata:
                        try:
                            movie_year = int(metadata['release_year'])
                            
                            # Check "before" constraint
                            if 'before' in temporal_constraints:
                                if movie_year < temporal_constraints['before']:
                                    is_entity_match = True
                            
                            # Check "after" constraint
                            elif 'after' in temporal_constraints:
                                if movie_year > temporal_constraints['after']:
                                    is_entity_match = True
                        except (ValueError, TypeError):
                            pass  # Skip if year is not a valid number
            
            # Categorize results
            if is_entity_match:
                entity_matches.append(c)
            else:
                non_entity_matches.append(c)
        
        # Sort each category by CE score
        entity_matches.sort(key=lambda s: (-s.score,
                                         -int(self.docs.get(s.id,{}).get("meta",{}).get("release_year",-1) or -1),
                                         str(self.docs.get(s.id,{}).get("title","")).lower()))
        
        non_entity_matches.sort(key=lambda s: (-s.score,
                                             -int(self.docs.get(s.id,{}).get("meta",{}).get("release_year",-1) or -1),
                                             str(self.docs.get(s.id,{}).get("title","")).lower()))
        
        # Combine: entity matches first (up to 3), then best semantic matches
        if has_entity_constraints and entity_matches:
            # Take top 3 entity matches, then fill with semantic matches
            result = entity_matches[:3] + non_entity_matches[:top_k-len(entity_matches[:3])]
            return result[:top_k]
        else:
            # No entity constraints, return normal CE ranking
            keep.sort(key=lambda s: (-s.score,
                                   -int(self.docs.get(s.id,{}).get("meta",{}).get("release_year",-1) or -1),
                                   str(self.docs.get(s.id,{}).get("title","")).lower()))
            return keep[:top_k]

    # ---------------- public ----------------
    def search(self, query: str, top_k: int = 50) -> List[SearchResult]:
        if not self.indexed:
            raise RuntimeError("index_data() first")
        cands = self._candidates(query)
        return self._ce_rerank(query, cands, top_k)
