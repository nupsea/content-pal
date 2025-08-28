from dataclasses import dataclass
import os
from opensearchpy import OpenSearch, RequestsHttpConnection


@dataclass(frozen=True)
class Cfg:
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embed_batch: int = int(os.getenv("EMBED_BATCH", "32"))
    index: str = os.getenv("OS_INDEX", "netflix_assets_v2")
    vector_dim: int = int(os.getenv("OS_VECTOR_DIM", "384"))
    url: str = os.getenv("OPENSEARCH_URL", "https://localhost:9200")
    user: str = os.getenv("OS_USER", "admin")
    pwd: str = os.getenv("OS_PASS", "admin")
    verify: bool = os.getenv("OS_VERIFY", 0) in (1, "1", "true", "True", "TRUE", True)
    timeout: int = int(os.getenv("OS_TIMEOUT", "60"))

def make_client(cfg: Cfg) -> OpenSearch:
    if cfg.url.startswith("https://"):
        return OpenSearch(
            cfg.url,
            http_auth=(cfg.user, cfg.pwd),
            verify_certs=cfg.verify,
            ssl_assert_hostname=cfg.verify,
            ssl_show_warn=cfg.verify,
            http_compress=True,
            connection_class=RequestsHttpConnection,
            timeout=cfg.timeout, max_retries=3, retry_on_timeout=True,
        )
    return OpenSearch(cfg.url, http_compress=True, timeout=cfg.timeout, max_retries=3, retry_on_timeout=True)