"""
Embedding utilities - Generate vector embeddings
Supports both API (OpenRouter) and local (SentenceTransformers) providers
Includes embedding cache for Cloud Run stateless deployments
"""
from typing import List, Optional, Dict, Any
import numpy as np
import hashlib
from datetime import datetime
import config
import os


class EmbeddingCache:
    """
    Embedding cache using LanceDB for persistent storage.
    Optimized for stateless Cloud Run deployments.
    """

    def __init__(
        self,
        db_path: str = None,
        table_name: str = "embedding_cache",
        storage_options: Optional[Dict[str, Any]] = None,
        embedding_dimension: int = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.table_name = table_name
        self.storage_options = storage_options
        self.embedding_dimension = embedding_dimension or getattr(config, 'EMBEDDING_DIMENSION', 4096)
        self.table = None
        self._enabled = getattr(config, 'ENABLE_EMBEDDING_CACHE', True)

        if self._enabled:
            self._init_cache()

    def _init_cache(self):
        """Initialize cache table in LanceDB."""
        try:
            import lancedb
            import pyarrow as pa

            is_cloud = self.db_path.startswith(("gs://", "s3://", "az://"))

            if is_cloud:
                self.db = lancedb.connect(self.db_path, storage_options=self.storage_options)
            else:
                os.makedirs(self.db_path, exist_ok=True)
                self.db = lancedb.connect(self.db_path)

            schema = pa.schema([
                pa.field("text_hash", pa.string()),
                pa.field("text_preview", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.embedding_dimension)),
                pa.field("is_query", pa.bool_()),
                pa.field("created_at", pa.string()),
                pa.field("hit_count", pa.int32())
            ])

            if self.table_name not in self.db.table_names():
                self.table = self.db.create_table(self.table_name, schema=schema)
                # Create scalar index on text_hash for O(1) lookups instead of full scan
                try:
                    self.table.create_scalar_index("text_hash", index_type="BTREE")
                    print(f"Created embedding cache table with BTREE index: {self.table_name}")
                except Exception as idx_err:
                    print(f"Created embedding cache table (index skipped): {self.table_name}")
            else:
                self.table = self.db.open_table(self.table_name)
                print(f"Opened embedding cache: {self.table.count_rows()} cached entries")

        except Exception as e:
            print(f"Warning: Failed to initialize embedding cache: {e}")
            self._enabled = False

    def _get_hash(self, text: str, is_query: bool) -> str:
        """Generate cache key from text."""
        normalized = text.strip().lower()
        key = f"{normalized}:{is_query}"
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, text: str, is_query: bool) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        if not self._enabled or self.table is None:
            return None

        try:
            text_hash = self._get_hash(text, is_query)
            results = self.table.search().where(
                f"text_hash = '{text_hash}'", prefilter=True
            ).limit(1).to_list()

            if results:
                return np.array(results[0]["embedding"], dtype=np.float32)
            return None

        except Exception as e:
            return None

    def put(self, text: str, embedding: np.ndarray, is_query: bool):
        """Store embedding in cache."""
        if not self._enabled or self.table is None:
            return

        try:
            text_hash = self._get_hash(text, is_query)

            existing = self.table.search().where(
                f"text_hash = '{text_hash}'", prefilter=True
            ).limit(1).to_list()

            if existing:
                return

            text_preview = text[:100] + "..." if len(text) > 100 else text

            self.table.add([{
                "text_hash": text_hash,
                "text_preview": text_preview,
                "embedding": embedding.tolist(),
                "is_query": is_query,
                "created_at": datetime.utcnow().isoformat(),
                "hit_count": 0
            }])

        except Exception as e:
            pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._enabled or self.table is None:
            return {"enabled": False}

        try:
            total = self.table.count_rows()
            return {
                "enabled": True,
                "total_entries": total,
                "table_name": self.table_name
            }
        except:
            return {"enabled": False}


class A0XEmbeddingProvider:
    """
    Embedding provider using a0x-models API.
    Uses intfloat/multilingual-e5-small (384D) optimized for semantic retrieval.
    Faster and more discriminative than larger models for this use case.
    """

    def __init__(
        self,
        api_url: str = None
    ):
        self.api_url = api_url or os.getenv("A0X_MODELS_URL", "https://a0x-models-679925931457.us-central1.run.app")
        self.dimension = 384  # intfloat/multilingual-e5-small
        self.model = "intfloat/multilingual-e5-small"
        print(f"a0x-models Embedding provider initialized: {self.model} ({self.dimension}D)")

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts using a0x-models API."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            import requests
            response = requests.post(
                f"{self.api_url}/embeddings",
                json={"texts": texts},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data["embeddings"]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"a0x-models API error: {e}")
            raise


class APIEmbeddingProvider:
    """
    Embedding provider using OpenRouter API.
    No cold start, better quality models.
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = "https://openrouter.ai/api/v1",
        model: str = None
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or config.OPENAI_API_KEY
        self.base_url = base_url
        self.model = model or getattr(config, 'EMBEDDING_MODEL', 'qwen/qwen3-embedding-8b')
        self.dimension = getattr(config, 'EMBEDDING_DIMENSION', 4096)

        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        print(f"API Embedding provider initialized: {self.model} ({self.dimension}D)")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using OpenRouter API."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )

            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)

        except Exception as e:
            print(f"API embedding error: {e}")
            raise


class LocalEmbeddingProvider:
    """
    Embedding provider using local SentenceTransformers.
    Free but has cold start penalty.
    """

    def __init__(self, model_name: str = None, use_optimization: bool = True):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization
        self.model = None
        self.dimension = None
        self.model_type = None
        self.supports_query_prompt = False

        self._init_model()

    def _init_model(self):
        """Initialize the local model."""
        print(f"Loading local embedding model: {self.model_name}")

        if self.model_name.lower().startswith("qwen"):
            self._init_qwen3_model()
        else:
            self._init_standard_model()

    def _init_qwen3_model(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer

            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B",
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }

            model_path = qwen3_models.get(self.model_name.lower(), self.model_name)

            if self.use_optimization:
                try:
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                except Exception:
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)

            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3"
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})

            print(f"Local model loaded: {self.dimension}D")

        except Exception as e:
            print(f"Failed to load Qwen3: {e}, falling back...")
            self._init_standard_model()

    def _init_standard_model(self):
        """Initialize standard SentenceTransformer model"""
        from sentence_transformers import SentenceTransformer

        fallback = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = self.model_name if not self.model_name.lower().startswith("qwen") else fallback

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.model_type = "sentence_transformer"
        self.supports_query_prompt = False

        print(f"SentenceTransformer loaded: {self.dimension}D")

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts using local model."""
        if isinstance(texts, str):
            texts = [texts]

        if self.supports_query_prompt and is_query:
            embeddings = self.model.encode(
                texts,
                prompt_name="query",
                show_progress_bar=False,
                normalize_embeddings=True
            )
        else:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True
            )

        return np.array(embeddings, dtype=np.float32)


class EmbeddingModel:
    """
    Unified embedding model with caching support.
    Supports a0x-models (384D), OpenRouter API (4096D), and local (SentenceTransformers) providers.

    Config options:
        EMBEDDING_PROVIDER = "a0x" | "api" | "local"
        EMBEDDING_MODEL = model name (optional, auto-detected for each provider)
        EMBEDDING_DIMENSION = auto-detected based on provider

    Provider comparison:
        a0x: 384D, fastest, best for retrieval, multilingual
        api: 4096D (OpenRouter), slower, general purpose
        local: varies, free but cold start
    """

    def __init__(
        self,
        model_name: str = None,
        provider: str = None,
        cache_db_path: str = None,
        storage_options: Optional[Dict[str, Any]] = None
    ):
        # Determine provider
        self.provider_type = provider or getattr(config, 'EMBEDDING_PROVIDER', 'a0x')
        self.model_name = model_name or config.EMBEDDING_MODEL

        # Initialize provider
        if self.provider_type == "a0x":
            self.provider = A0XEmbeddingProvider()
            self.dimension = self.provider.dimension
        elif self.provider_type == "api":
            self.provider = APIEmbeddingProvider(model=self.model_name)
            self.dimension = self.provider.dimension
        else:
            self.provider = LocalEmbeddingProvider(model_name=self.model_name)
            self.dimension = self.provider.dimension

        # Initialize cache
        enable_cache = getattr(config, 'ENABLE_EMBEDDING_CACHE', True)
        if enable_cache:
            self.cache = EmbeddingCache(
                db_path=cache_db_path,
                storage_options=storage_options,
                embedding_dimension=self.dimension
            )
        else:
            self.cache = None

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts to vectors with caching support.

        Args:
            texts: List of texts to encode
            is_query: Whether these are query texts (for prompt optimization)
        """
        if isinstance(texts, str):
            texts = [texts]

        if not self.cache:
            return self._compute_embeddings(texts, is_query)

        results = [None] * len(texts)
        texts_to_compute = []
        indices_to_compute = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text, is_query)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)

        if texts_to_compute:
            computed = self._compute_embeddings(texts_to_compute, is_query)
            for idx, text, emb in zip(indices_to_compute, texts_to_compute, computed):
                self.cache.put(text, emb, is_query)
                results[idx] = emb

        return np.array(results)

    def _compute_embeddings(self, texts: List[str], is_query: bool) -> np.ndarray:
        """Compute embeddings using the configured provider."""
        if self.provider_type == "api":
            return self.provider.encode(texts)
        else:
            return self.provider.encode(texts, is_query=is_query)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encode single text."""
        return self.encode([text], is_query=is_query)[0]

    def encode_query(self, queries: List[str]) -> np.ndarray:
        """Encode queries with optimal settings."""
        return self.encode(queries, is_query=True)

    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents."""
        return self.encode(documents, is_query=False)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}
