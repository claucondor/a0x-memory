# Prompt para continuar sesión

Copia y pega esto cuando inicies una nueva sesión de Claude:

---

Estamos trabajando en optimizaciones para a0x-memory. Lee el archivo `CLAUDE_SESSION_STATE.md` para ver el contexto completo.

**Estado actual:**
1. Hay un commit local pendiente de push (`3823c53`)
2. Cambiamos LLM a llama-3.1-8b y embeddings a API (qwen3-embedding-8b 4096D)
3. La red estaba fallando (no conectaba a OpenRouter)

**Tareas pendientes:**
1. Hacer `git push origin main`
2. Probar que los API embeddings funcionen
3. Borrar datos viejos de GCS si es necesario (`gs://a0x-memory/lancedb`)
4. Continuar con Hybrid Search + Fusion (Fase 2 del plan)

Lee `CLAUDE_SESSION_STATE.md` y `MCP/RETRIEVAL_OPTIMIZATION_PLAN.md` para más detalles. Primero verifica la red y haz el push.
