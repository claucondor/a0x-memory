# Plan de Optimización del Sistema de Retrieval - SimpleMem

## Resumen Ejecutivo

Basado en investigación de mejores prácticas en RAG 2025, este plan propone mejoras al sistema de retrieval de SimpleMem para:
- **Reducir costos** eliminando LLM calls innecesarios en búsqueda
- **Mejorar calidad** con hybrid search nativo + reranking
- **Aumentar escalabilidad** usando índices FTS de LanceDB

## Estado Actual vs Propuesto

| Aspecto | Actual | Propuesto | Mejora |
|---------|--------|-----------|--------|
| Keyword Search | Carga TODO a pandas, itera | FTS Index nativo LanceDB | ~100x más rápido |
| Hybrid Fusion | Manual | RRF/Linear Combination | Mejor ranking |
| LLM en búsqueda | 2-4 calls (planning) | 0 calls (default) | ~$0.0001 ahorro/query |
| Reranking | Ninguno | Opcional (Jina/Cohere) | +15-30% accuracy |

## Arquitectura Propuesta

```
Query → [Embedding] → Parallel Search → [Fusion] → [Reranker] → Results
              │              │               │           │
              │       ┌──────┴──────┐        │           │
              │       │             │        │           │
              └───► Vector     FTS Index ────┘           │
                   Search      (BM25)                    │
                     │             │                     │
                     └─────────────┘                     │
                            │                            │
                      RRF/Linear ────────────────────────┘
                      Combination        (opcional)
```

## Fases de Implementación

### Fase 1: FTS Index Nativo (Prioridad Alta)
**Impacto: Escalabilidad + Performance**

Reemplazar keyword search manual con FTS index de LanceDB:

```python
# Crear índice FTS al inicializar tabla
table.create_fts_index("lossless_restatement", replace=True)
table.create_fts_index("keywords", replace=True)  # Campo array

# Búsqueda FTS nativa (usa Tantivy/BM25)
results = table.search("query terms", query_type="fts").limit(10).to_list()
```

**Beneficios:**
- No carga datos a memoria
- BM25 scoring nativo
- Soporte stemming (en_stem, es_stem)
- ~100x más rápido que iteración manual

**Archivos a modificar:**
- `server/database/vector_store.py`: Agregar create_fts_index(), usar search nativo
- `config/settings.py`: Agregar fts_language config

### Fase 2: Hybrid Search con Fusion (Prioridad Alta)
**Impacto: Mejor calidad de retrieval**

LanceDB soporta hybrid search nativo con rerankers:

```python
from lancedb.rerankers import RRFReranker, LinearCombinationReranker

# Opción 1: RRF (Reciprocal Rank Fusion) - Default recomendado
reranker = RRFReranker(k=60)  # k controla peso de posición

# Opción 2: Linear Combination - Más control
reranker = LinearCombinationReranker(weight=0.7)  # 70% vector, 30% FTS

# Hybrid search en una línea
results = (
    table.search(query_embedding, query_type="hybrid")
    .rerank(reranker=reranker)
    .limit(top_k)
    .to_list()
)
```

**Beneficios según investigación:**
- Hybrid mejora NDCG 26-31% vs dense-only (arXiv:2402.03367)
- RRF combina rankings sin necesidad de normalizar scores
- Linear permite ajustar peso según caso de uso

**Configuración recomendada:**
```python
# Para memoria de agentes (más factual):
reranker = LinearCombinationReranker(weight=0.6)  # 60% semantic, 40% keyword

# Para búsqueda general:
reranker = RRFReranker(k=60)
```

### Fase 3: Deshabilitar Planning por Default (Prioridad Alta)
**Impacto: Reducción de costos**

El "planning" actual usa 2-4 LLM calls por query:
1. `_analyze_information_requirements` - analiza complejidad
2. `_generate_targeted_queries` - genera sub-queries
3. `_check_completeness` - verifica si hay info faltante
4. `_generate_missing_info_queries` - genera queries adicionales

**Costo actual:** ~$0.0001-0.0003 por query (con llama-3.1-8b)

**Propuesta:**
```python
# En config/settings.py
enable_planning: bool = False  # Default OFF (era True)
enable_reflection: bool = False  # Default OFF

# El usuario puede habilitarlo para queries complejas:
# LLM_ENABLE_PLANNING=true
```

**Cuándo usar planning:**
- Queries multi-hop complejas
- Cuando se requiere alta precisión
- Casos donde el costo extra vale la pena

### Fase 4: Reranker Opcional (Prioridad Media)
**Impacto: +15-30% accuracy cuando se necesita**

Opciones de rerankers externos:

| Reranker | Costo | Latencia | Calidad |
|----------|-------|----------|---------|
| **Jina v2** | ~$0.02/1M tokens | ~200ms | Excelente |
| **Cohere v3.5** | ~$1/1K queries | ~600ms | Mejor |
| **ColBERT** | Self-hosted | ~50ms | Muy buena |

**Implementación con LanceDB:**
```python
from lancedb.rerankers import CohereReranker, CrossEncoderReranker

# Cohere (API)
reranker = CohereReranker(api_key="...", model_name="rerank-multilingual-v3.0")

# Cross-encoder local (gratis pero más lento)
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

# Uso
results = table.search(query, query_type="hybrid").rerank(reranker=reranker).to_list()
```

**Recomendación:**
- Default: Sin reranker externo (RRF interno suficiente)
- Opcional: Jina v2 para casos que requieren máxima precisión

### Fase 5: Structured Filters Optimizados (Prioridad Baja)
**Impacto: Búsquedas por metadatos más rápidas**

LanceDB soporta filtros SQL-like eficientes:

```python
# En lugar de cargar todo y filtrar en Python:
results = (
    table.search(query_embedding)
    .where("persons LIKE '%Carlos%'")
    .where("timestamp >= '2025-01-01'")
    .limit(10)
    .to_list()
)
```

## Resumen de Cambios por Archivo

### `config/settings.py`
```python
# Nuevas configuraciones
enable_planning: bool = False  # Cambiar default
enable_reflection: bool = False
fts_language: str = "en_stem"  # Stemming para FTS
hybrid_weight: float = 0.7  # Peso vector vs FTS
use_external_reranker: bool = False
reranker_provider: str = "jina"  # jina, cohere, local
```

### `server/database/vector_store.py`
```python
# Agregar métodos:
def create_fts_indexes(self, table_name: str)
def hybrid_search_native(self, table_name: str, query: str, query_embedding: List[float], top_k: int)

# Modificar:
def keyword_search()  # Usar FTS nativo
```

### `server/core/retriever.py`
```python
# Simplificar retrieve() para usar hybrid nativo
# Mover planning a método separado opcional
# Agregar soporte para reranker externo
```

## Estimación de Impacto

### Costos por Query (Retrieval)

| Configuración | Costo | Latencia |
|---------------|-------|----------|
| **Actual (planning ON)** | ~$0.0002 | ~2-5s |
| **Propuesto (planning OFF)** | ~$0.00001 | ~100-300ms |
| **Con Jina reranker** | ~$0.00003 | ~400-600ms |

### Calidad de Retrieval

| Método | NDCG Estimado | Recall@10 |
|--------|---------------|-----------|
| Solo vector | Baseline | ~70% |
| Hybrid (RRF) | +26-31% | ~85% |
| Hybrid + Reranker | +40-50% | ~92% |

## Plan de Ejecución

```
Semana 1: Fase 1 + 3 (FTS Index + Disable Planning)
├── Implementar FTS index en vector_store.py
├── Cambiar defaults en settings.py
└── Tests de performance

Semana 2: Fase 2 (Hybrid Search Nativo)
├── Integrar RRF/Linear rerankers de LanceDB
├── Refactorizar retriever.py
└── Tests de calidad (recall, precision)

Semana 3: Fase 4 (Reranker Opcional)
├── Agregar integración Jina/Cohere
├── Configuración por environment
└── Benchmarks comparativos

Semana 4: Fase 5 + Documentación
├── Optimizar structured filters
├── Documentar configuraciones
└── PR a upstream SimpleMem
```

## Referencias

- [LanceDB Full-Text Search](https://docs.lancedb.com/search/full-text-search)
- [LanceDB Hybrid Search](https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/)
- [Hybrid Search + Reranking Report](https://blog.lancedb.com/hybrid-search-and-reranking-report/)
- [arXiv:2402.03367 - Hybrid Retrieval](https://arxiv.org/abs/2402.03367)
- [Jina Reranker v2](https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search/)
- [ColBERT Late Interaction](https://github.com/stanford-futuredata/ColBERT)
- [Best Reranking Models 2025](https://www.zeroentropy.dev/articles/ultimate-guide-to-choosing-the-best-reranking-model-in-2025)
