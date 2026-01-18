# Estado de Sesión - Optimización a0x-memory

**Fecha:** 2026-01-14
**Problema actual:** Red/WSL no conecta a OpenRouter (timeout en TLS handshake)

---

## Commits pendientes de push

```bash
git push origin main
```

Commit local: `3823c53` - perf: optimize LLM and embedding providers for cost and latency

---

## Cambios realizados en esta sesión

### 1. LLM Model (HECHO)
- **Antes:** `gpt-4.1-mini`
- **Después:** `meta-llama/llama-3.1-8b-instruct`
- **Beneficio:** 5.7x más rápido, 14x más barato (probado con test_planning_models.py)

### 2. Embedding Provider (HECHO)
- **Antes:** Local SentenceTransformer (Qwen3-0.6B, 1024D)
- **Después:** API OpenRouter (qwen/qwen3-embedding-8b, 4096D)
- **Archivo:** `utils/embedding.py` - ahora soporta `EMBEDDING_PROVIDER = "api" | "local"`
- **Beneficio:** Sin cold start, mejor calidad, $0.02/1M tokens

### 3. Workers optimizados (HECHO)
- **Antes:** `MAX_PARALLEL_WORKERS = 16`, `MAX_RETRIEVAL_WORKERS = 8`
- **Después:** Ambos = 4 (optimizado para Cloud Run 2-4 vCPUs)

### 4. MCP marcado deprecated (HECHO)
- Archivo: `MCP/DEPRECATED.md`
- Razón: Código duplicado sin las optimizaciones del root

---

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `utils/embedding.py` | APIEmbeddingProvider + LocalEmbeddingProvider + cache |
| `config.py.example` | Nuevos defaults (llama, api embedding, workers) |
| `config.py` | Tu config local (en .gitignore) |
| `tests/test_planning_models.py` | Test para comparar modelos LLM |
| `MCP/DEPRECATED.md` | Aviso de deprecación |

---

## Pendiente por hacer

### Inmediato (cuando la red funcione)
1. [ ] `git push origin main` - pushear commit local
2. [ ] Probar API embeddings con OpenRouter
3. [ ] Borrar datos de GCS si es necesario: `gs://a0x-memory/lancedb`

### Plan de optimización restante
1. [ ] **Hybrid Search + Fusion** - RRF/Linear Combination (Fase 2)
2. [ ] **IVF-PQ Index Tuning** - Mejor recall/latencia
3. [ ] Structured Filters optimizados

---

## Config actual (config.py)

```python
# LLM
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"

# Embeddings
EMBEDDING_PROVIDER = "api"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
EMBEDDING_DIMENSION = 4096

# Workers
MAX_PARALLEL_WORKERS = 4
MAX_RETRIEVAL_WORKERS = 4

# Planning (mantener activado, con llama es rápido)
ENABLE_PLANNING = True
ENABLE_REFLECTION = True
```

---

## Comandos útiles para retomar

```bash
# Ver estado git
git status
git log --oneline -5

# Push cuando la red funcione
git push origin main

# Probar embeddings
export OPENROUTER_API_KEY=sk-or-v1-...
python -c "from utils.embedding import EmbeddingModel; m = EmbeddingModel(); print(m.encode_single('test').shape)"

# Probar planning con diferentes modelos
python tests/test_planning_models.py --all-queries
```

---

## Resultados del benchmark de modelos

| Modelo | Latencia Avg | Costo Total |
|--------|-------------|-------------|
| **llama-3.1-8b** | **1.17s** | **$0.00017** |
| gemini-2.0-flash-lite | 4.61s | $0.00067 |
| gpt-4.1-mini | 6.66s | $0.00245 |
| qwen-2.5-7b | 6.07s | $0.00013 |
| mistral-small-3 | 10.24s | $0.00020 |

**Conclusión:** llama-3.1-8b es el mejor para planning/reflection
