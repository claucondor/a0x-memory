# Ingestion Pipeline Design

Arquitectura para ingesta de mensajes en Cloud Run (stateless).

---

## Problema

1. Cloud Run es stateless - instancias pueden morir en cualquier momento
2. Múltiples instancias pueden estar corriendo
3. Procesar mensajes con LLM es lento (~2-4s por batch)
4. No queremos perder mensajes si el proceso falla
5. Necesitamos deduplicación (mismo mensaje puede llegar múltiples veces)

---

## Arquitectura: Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION LAYER                              │
│                     (Fast, No LLM)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
     Twitter API              │            Telegram Bot
     Farcaster Hub            │            Discord Bot
           │                  │                  │
           ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAW MESSAGES TABLE                           │
│                     (Short-term Memory)                          │
│                                                                  │
│  - Insert inmediato (< 10ms)                                    │
│  - Sin procesamiento LLM                                         │
│  - Deduplicación por message_id                                  │
│  - Buffer para procesamiento posterior                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │  Background Worker / Cron
                              │  (batch processing)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                             │
│                     (LLM, Embeddings)                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORIES TABLE                               │
│                     (Long-term Memory)                           │
│                                                                  │
│  - Atomic facts extraídos por LLM                               │
│  - Embeddings para semantic search                               │
│  - Privacy scope aplicado                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Schema: Raw Messages

Buffer de mensajes sin procesar. Inserciones rápidas, sin LLM.

```python
raw_messages = {
    # Identity
    "message_id": str,           # PK - platform-specific unique ID
    "platform": str,             # "telegram" | "twitter" | "farcaster" | "discord"

    # Context
    "agent_id": str,             # which agent received this
    "user_id": str | None,       # internal user_id if known
    "platform_user_id": str,     # platform-specific user ID
    "platform_username": str,    # @handle

    # Group context
    "group_id": str | None,      # null = DM
    "group_name": str | None,
    "context_type": str,         # "dm" | "group" | "public_timeline"

    # Message content
    "content": str,              # raw message text
    "reply_to_message_id": str | None,
    "attachments": List[str],    # URLs to media if any

    # Timestamps
    "platform_timestamp": str,   # when message was sent (from platform)
    "received_at": str,          # when we received it

    # Processing status
    "status": str,               # "pending" | "processing" | "processed" | "failed" | "skipped"
    "processed_at": str | None,
    "error": str | None,         # if failed, why
    "memory_ids": List[str],     # IDs of memories created from this message
}
```

**Índices:**
- Primary: `message_id` (dedup)
- Query: `agent_id + status` (get pending for processing)
- Query: `agent_id + group_id + platform_timestamp` (conversation order)
- Query: `platform_user_id` (find all messages from user)

---

## Processing Flows

### Flow 1: DM Message

```
1. Telegram webhook receives message
2. API endpoint:
   - Extract message data
   - Lookup/create user_id from platform_user_id
   - Insert into raw_messages (status="pending")
   - Return 200 OK immediately

3. Background worker (every 30s or on trigger):
   - Query: raw_messages WHERE agent_id=X AND status="pending" ORDER BY received_at
   - Group by conversation (user_id)
   - For each conversation batch:
     a. Get recent context (last N processed messages)
     b. Send to LLM for memory extraction (modelo TBD)
     c. Insert memories with privacy_scope="private"
     d. Update raw_messages.status="processed"
```

### Flow 2: Group Message

```
1. Telegram webhook receives group message
2. API endpoint:
   - Same as DM but with group_id set
   - Insert into raw_messages (status="pending")
   - Return 200 OK immediately

3. Background worker:
   - Query: raw_messages WHERE agent_id=X AND group_id=Y AND status="pending"
   - Check: was agent mentioned? is message relevant?
     - If not relevant AND group config says only_when_mentioned:
       - Update status="skipped"
       - Continue
   - If relevant:
     - Get group context (topic, recent messages)
     - Send to LLM for memory extraction
     - Insert memories with privacy_scope="group_only"
     - Update raw_messages.status="processed"
```

### Flow 3: Batch Catchup (Group History)

```
1. Agent joins new group
2. Fetch last N messages from Telegram API
3. Bulk insert into raw_messages (status="pending")
4. Background worker processes in chronological order
5. Build initial group context
```

---

## Short-term vs Long-term Memory

| Aspect | Raw Messages (Short-term) | Memories (Long-term) |
|--------|---------------------------|----------------------|
| Content | Exact message text | Extracted atomic facts |
| Insert speed | < 10ms | ~2-4s (LLM + embedding) |
| Search | By ID, timestamp | Semantic + keyword |
| Retention | 7-30 days | Configurable per type |
| Purpose | Buffer, audit trail | Knowledge retrieval |

### Using Short-term for Context

Cuando el agente necesita responder, puede usar ambas:

```python
def get_context_for_response(agent_id, user_id, group_id=None):
    # 1. Recent raw messages (last 10, for immediate context)
    recent_raw = raw_messages.filter(
        agent_id=agent_id,
        group_id=group_id,
        platform_timestamp > now() - 5_minutes
    ).order_by(platform_timestamp).limit(10)

    # 2. Relevant long-term memories
    memories = semantic_search(
        query=current_message,
        agent_id=agent_id,
        user_id=user_id,
        group_id=group_id,
        # respect privacy_scope
    )

    # 3. Combine for LLM context
    return {
        "recent_conversation": recent_raw,
        "relevant_memories": memories,
        "user_summary": get_user_summary(user_id),
        "group_context": get_group_context(group_id) if group_id else None
    }
```

---

## Deduplication Strategy

### Por qué ocurren duplicados
- Webhook retry (Telegram reintenta si no recibe 200)
- Multiple bot instances
- Manual replay/recovery

### Estrategia
```python
def ingest_message(message):
    message_id = f"{platform}:{platform_message_id}"

    # Check if exists
    existing = raw_messages.get(message_id)
    if existing:
        return {"status": "duplicate", "message_id": message_id}

    # Insert
    raw_messages.insert({
        "message_id": message_id,
        ...
    })
    return {"status": "created", "message_id": message_id}
```

---

## Cloud Run Considerations

### Stateless Processing
```
- No in-memory state between requests
- Database is source of truth
- Worker can run on any instance
```

### Concurrency Control
```python
# Prevent multiple workers processing same message
def process_pending_messages(agent_id, batch_size=10):
    # Atomic: SELECT + UPDATE status to "processing"
    messages = raw_messages.filter(
        agent_id=agent_id,
        status="pending"
    ).limit(batch_size).update(status="processing")

    for msg in messages:
        try:
            process_message(msg)
            msg.update(status="processed")
        except Exception as e:
            msg.update(status="failed", error=str(e))
```

### Worker Trigger Options

| Option | Pros | Cons |
|--------|------|------|
| Cloud Scheduler (cron) | Simple, reliable | Fixed interval, min 1 min |
| Pub/Sub on insert | Real-time | More complex |
| Cloud Tasks | Delayed processing | More complex |
| Same request (async) | Simple | Blocks response if slow |

**Recommended:** Cloud Scheduler every 30s + Pub/Sub for high-priority (DMs)

---

## Retention & Cleanup

### Raw Messages
```python
# Cron job: delete old processed messages
DELETE FROM raw_messages
WHERE status IN ("processed", "skipped")
AND received_at < now() - 7_days
```

### Failed Messages
```python
# Keep failed for debugging, cleanup after 30 days
DELETE FROM raw_messages
WHERE status = "failed"
AND received_at < now() - 30_days
```

---

## API Endpoints

### Ingest (Fast Path)
```
POST /ingest/telegram
POST /ingest/twitter
POST /ingest/farcaster

Body: { platform-specific webhook payload }
Response: { "status": "created|duplicate", "message_id": "..." }
Latency target: < 50ms
```

### Process Trigger (Worker)
```
POST /process/trigger
Body: { "agent_id": "...", "batch_size": 10 }
Response: { "processed": 5, "skipped": 2, "failed": 0 }
Called by: Cloud Scheduler or Pub/Sub
```

### Status (Debug)
```
GET /ingest/status?agent_id=X
Response: {
  "pending": 15,
  "processing": 2,
  "processed_today": 450,
  "failed_today": 3
}
```

---

## LLM Integration Points (Modelo TBD)

| Step | Purpose | Input | Output |
|------|---------|-------|--------|
| Memory extraction | Convert message to atomic facts | Message + context | List of memories |
| Relevance check | Is group message worth processing? | Message + group topic | bool |
| User intent | Is this a command? Question? Statement? | Message | intent classification |

---

## Open Questions

1. **¿Cuánto raw message history mantener?**
   - 7 días? 30 días? Solo pending?

2. **¿Processing order?**
   - FIFO estricto por timestamp?
   - Priority queue (DMs > Groups)?

3. **¿Qué hacer si LLM falla repetidamente?**
   - Retry con backoff?
   - Skip después de N intentos?
   - Alert?

4. **¿Batch size óptimo para LLM?**
   - 1 mensaje = más contexto específico
   - 10 mensajes = más eficiente pero menos contexto

5. **¿Raw messages en LanceDB o DB separada?**
   - LanceDB: simple, todo junto
   - PostgreSQL: mejor para queries transaccionales
   - Considerar: volumen, queries, costo

---

## Next Steps

- [ ] Decidir storage para raw_messages (LanceDB vs Postgres vs Firestore)
- [ ] Implementar raw_messages schema
- [ ] Crear endpoints de ingestion
- [ ] Implementar worker de procesamiento
- [ ] Configurar Cloud Scheduler
- [ ] Definir retention policies
- [ ] Métricas y alertas
