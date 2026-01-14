"""
Embedding utilities - Generate vector embeddings using SentenceTransformers
Supports Qwen3 Embedding models through SentenceTransformers interface
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
        storage_options: Optional[Dict[str, Any]] = None
    ):
        self.db_path = db_path or config.LANCEDB_PATH
        self.table_name = table_name
        self.storage_options = storage_options
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
                pa.field("embedding", pa.list_(pa.float32())),
                pa.field("is_query", pa.bool_()),
                pa.field("created_at", pa.string()),
                pa.field("hit_count", pa.int32())
            ])

            if self.table_name not in self.db.table_names():
                self.table = self.db.create_table(self.table_name, schema=schema)
                print(f"Created embedding cache table: {self.table_name}")
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


class EmbeddingModel:
    """
    Embedding model using SentenceTransformers (supports Qwen3 and other models)
    Includes embedding cache for repeated queries/documents
    """

    def __init__(
        self,
        model_name: str = None,
        use_optimization: bool = True,
        cache_db_path: str = None,
        storage_options: Optional[Dict[str, Any]] = None
    ):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.use_optimization = use_optimization

        enable_cache = getattr(config, 'ENABLE_EMBEDDING_CACHE', True)
        if enable_cache:
            self.cache = EmbeddingCache(
                db_path=cache_db_path,
                storage_options=storage_options
            )
        else:
            self.cache = None

        print(f"Loading embedding model: {self.model_name}")
        
        # Check if it's a Qwen3 model (through SentenceTransformers)
        if self.model_name.startswith("qwen3"):
            self._init_qwen3_sentence_transformer()
        else:
            self._init_standard_sentence_transformer()

    def _init_qwen3_sentence_transformer(self):
        """Initialize Qwen3 model using SentenceTransformers"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Map model names to actual model paths
            qwen3_models = {
                "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
                "qwen3-4b": "Qwen/Qwen3-Embedding-4B", 
                "qwen3-8b": "Qwen/Qwen3-Embedding-8B"
            }
            
            model_path = qwen3_models.get(self.model_name, self.model_name)
            print(f"Loading Qwen3 model via SentenceTransformers: {model_path}")
            
            # Initialize with optimization settings
            if self.use_optimization:
                try:
                    # Try to use flash_attention_2 and left padding for better performance
                    self.model = SentenceTransformer(
                        model_path,
                        model_kwargs={
                            "attn_implementation": "flash_attention_2", 
                            "device_map": "auto"
                        },
                        tokenizer_kwargs={"padding_side": "left"},
                        trust_remote_code=True
                    )
                    print("Qwen3 loaded with flash_attention_2 optimization")
                except Exception as e:
                    print(f"Flash attention failed ({e}), using standard loading...")
                    self.model = SentenceTransformer(model_path, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_path, trust_remote_code=True)
            
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "qwen3_sentence_transformer"
            
            # Check if Qwen3 supports query prompts
            self.supports_query_prompt = hasattr(self.model, 'prompts') and 'query' in getattr(self.model, 'prompts', {})
            
            print(f"Qwen3 model loaded successfully with dimension: {self.dimension}")
            if self.supports_query_prompt:
                print("Query prompt support detected")
                
        except Exception as e:
            print(f"Failed to load Qwen3 model: {e}")
            print("Falling back to default SentenceTransformers model...")
            self._fallback_to_sentence_transformer()

    def _init_standard_sentence_transformer(self):
        """Initialize standard SentenceTransformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            self.model_type = "sentence_transformer"
            self.supports_query_prompt = False
            print(f"SentenceTransformer model loaded with dimension: {self.dimension}")
        except Exception as e:
            print(f"Failed to load SentenceTransformer model: {e}")
            raise

    def _fallback_to_sentence_transformer(self):
        """Fallback to default SentenceTransformer model"""
        fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"Using fallback model: {fallback_model}")
        self.model_name = fallback_model
        self._init_standard_sentence_transformer()

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode list of texts to vectors with caching support

        Args:
        - texts: List of texts to encode
        - is_query: Whether these are query texts (for Qwen3 prompt optimization)
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
        """Compute embeddings without cache."""
        if self.model_type == "qwen3_sentence_transformer" and self.supports_query_prompt and is_query:
            return self._encode_with_query_prompt(texts)
        else:
            return self._encode_standard(texts)

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """
        Encode single text
        
        Args:
        - text: Text to encode
        - is_query: Whether this is a query text (for Qwen3 prompt optimization)
        """
        return self.encode([text], is_query=is_query)[0]
    
    def encode_query(self, queries: List[str]) -> np.ndarray:
        """
        Encode queries with optimal settings for Qwen3
        """
        return self.encode(queries, is_query=True)
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """
        Encode documents (no query prompt)
        """
        return self.encode(documents, is_query=False)
    
    def _encode_with_query_prompt(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Qwen3 query prompt"""
        try:
            embeddings = self.model.encode(
                texts, 
                prompt_name="query",  # Use Qwen3's query prompt
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
        except Exception as e:
            print(f"Query prompt encoding failed: {e}, falling back to standard encoding")
            return self._encode_standard(texts)
    
    def _encode_standard(self, texts: List[str]) -> np.ndarray:
        """Encode texts using standard method"""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"enabled": False}
