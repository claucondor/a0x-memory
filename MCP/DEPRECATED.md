# MCP Server - DEPRECATED

> **Status**: DEPRECATED - No usar en producción hasta actualizar
> **Fecha**: 2026-01-14
> **Usar en su lugar**: FastAPI server (`api.py`)

## Problema

El MCP server tiene su propia implementación separada del código principal.
Las optimizaciones recientes NO están aplicadas al MCP:

### Optimizaciones en ROOT (api.py) que MCP NO tiene:

| Optimización | Archivo ROOT | Archivo MCP (sin actualizar) |
|--------------|--------------|------------------------------|
| LanceDB native search (no to_pandas) | `database/vector_store.py` | `MCP/server/database/vector_store.py` |
| Embedding cache | `utils/embedding.py` | No existe en MCP |
| FTS index nativo | `database/vector_store.py` | No existe en MCP |
| Structured search SQL | `database/vector_store.py` | No existe en MCP |

### Diferencias de arquitectura:

```
ROOT (USAR ESTE)                    MCP (DEPRECATED)
├── database/vector_store.py        ├── server/database/vector_store.py (copia vieja)
├── utils/embedding.py (+ cache)    ├── integrations/openrouter.py (solo API)
├── core/hybrid_retriever.py        ├── server/core/retriever.py (diferente)
├── core/memory_builder.py          ├── server/core/memory_builder.py (diferente)
└── api.py (FastAPI)                └── http_server.py (Hono-style)
```

## TODO para reactivar MCP

Si se necesita MCP en el futuro, hacer refactor para que importe del ROOT:

1. [ ] `MCP/server/database/vector_store.py` → importar de `database.vector_store`
2. [ ] `MCP/server/core/retriever.py` → importar de `core.hybrid_retriever`
3. [ ] `MCP/server/core/memory_builder.py` → importar de `core.memory_builder`
4. [ ] Agregar `EmbeddingProvider` configurable (local vs API)
5. [ ] Unificar configs: `config.py` ↔ `MCP/config/settings.py`
6. [ ] Tests compartidos

## Servidor recomendado

```bash
# Usar FastAPI (con todas las optimizaciones)
uvicorn api:app --host 0.0.0.0 --port 8000

# NO usar MCP server hasta actualizar
# python -m MCP.run  # DEPRECATED
```

## Commits con optimizaciones (solo en ROOT)

- `8b78df2` - feat: add embedding cache for Cloud Run
- `5517182` - perf: replace to_pandas() with native LanceDB
- `02b11e3` - feat: optimize for cost with llama-3.1-8b and GCS
