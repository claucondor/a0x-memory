# A0X Memory API Reference

## Overview

Sistema de memoria para AI agents con soporte para:
- **DM Memories**: Conversaciones privadas 1-on-1
- **Group Memories**: Conocimiento compartido del grupo
- **User Memories**: Lo que cada usuario ha dicho en grupos
- **User Profiles**: Perfil global del usuario (cross-context)
- **Group Profiles**: Perfil del grupo
- **User Facts**: Hechos verificados sobre usuarios (evidence-based)
- **Summaries**: Resúmenes jerárquicos (micro/chunk/block)

## Privacy Model

| Contexto | El agente VE | El agente NO VE |
|----------|--------------|-----------------|
| **DM con User X** | DM history con X, Todo lo que X dijo en grupos, Facts de X, CrossGroupMemories de X | DMs de otros users, Lo que OTROS dijeron en grupos |
| **Group A (Speaker=X)** | History de Group A, Memorias de Group A, X's shareable DM memories, Facts de X, Profiles de involved_users | DMs de otros, Memorias de otros grupos, DMs no-shareable de X |

---

## Endpoints

### 1. Health & System

#### `GET /health`
**Descripción**: Health check del servicio.

```bash
curl http://136.118.160.81:8080/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "memory_instances": 1,
  "timestamp": "2026-01-30T02:48:09.025841"
}
```

---

### 2. Memory Ingestion

#### `POST /v1/memory/passive`
**Descripción**: Agregar mensaje de forma pasiva (fire-and-forget). El sistema decide cuándo procesar.

**Cuándo usar**: Cada mensaje de usuario en Telegram/XMTP/etc.

**Parámetros**:
```json
{
  "agent_id": "jessexbt",
  "message": "I'm a Solidity developer with 3 years experience",
  "platform_identity": {
    "platform": "telegram",
    "telegramId": 123456,
    "username": "alice_dev",
    "chatId": "-100001"  // Negativo = grupo, null/ausente = DM
  },
  "speaker": "alice_dev"
}
```

**Comportamiento**:
- `chatId` negativo → Mensaje de GRUPO → Genera `group_memories`, `user_memories`, `interaction_memories`
- `chatId` null/ausente → Mensaje de DM → Genera `dm_memories` con `is_shareable` decidido por LLM

**Response**:
```json
{
  "success": true,
  "is_group": false,
  "group_id": null,
  "user_id": "telegram:123456",
  "processing_scheduled": true,
  "memories_created": 0,
  "is_spam": false,
  "is_blocked": false,
  "spam_score": 0.0
}
```

**Preguntas que habilita**:
- "¿Qué sabe hacer este usuario?" → Extrae expertise de sus mensajes
- "¿De qué se habló en el grupo?" → Genera group_memories

---

#### `POST /v1/memory/active`
**Descripción**: Agregar mensaje Y obtener contexto inmediatamente (para responder).

**Cuándo usar**: Cuando el agente necesita responder al mensaje.

**Parámetros**:
```json
{
  "agent_id": "jessexbt",
  "message": "What grants are available for DeFi projects?",
  "platform_identity": {
    "platform": "telegram",
    "telegramId": 123456,
    "username": "alice_dev",
    "chatId": "-100001"
  },
  "speaker": "alice_dev",
  "involved_users": ["telegram:123456", "telegram:789012"]
}
```

**Response**: Incluye `formatted_context` listo para el LLM con:
- Recent messages (Firestore window)
- Group memories
- User memories
- Speaker's shareable DM memories (si es grupo)
- User profiles de involved_users

---

### 3. Memory Retrieval (Context)

#### `POST /v1/memory/context`
**Descripción**: Obtener contexto para una query SIN agregar mensaje.

**Cuándo usar**: RAG puro, cuando ya tienes el mensaje y solo necesitas contexto.

**Parámetros**:
```json
{
  "agent_id": "jessexbt",
  "query": "What does Elena specialize in?",
  "platform_identity": {
    "platform": "telegram",
    "telegramId": 88001,
    "username": "elena_dev",
    "chatId": "-100002"
  },
  "involved_users": ["telegram:88001"],
  "include_recent": true,
  "recent_limit": 10,
  "memory_limit": 5
}
```

**Response**:
```json
{
  "success": true,
  "recent_messages": [...],
  "relevant_memories": [...],
  "user_profile": [...],
  "group_profile": {...},
  "formatted_context": "## Group Knowledge\n1. ...\n\n## Speaker's Personal Context (shareable)\n1. Elena specializes in TypeScript..."
}
```

**Preguntas que responde**:
- "¿Qué sabe hacer Elena?" → Busca en group_memories + speaker_dm_memories
- "¿De qué se habló ayer?" → Busca en group_memories con temporal scoring
- "¿Qué dijeron sobre el proyecto X?" → Busca semánticamente en todas las tablas

---

#### `GET /v1/memory/stats/{agent_id}`
**Descripción**: Estadísticas del agente.

```bash
curl http://136.118.160.81:8080/v1/memory/stats/jessexbt
```

**Response**:
```json
{
  "agent_id": "jessexbt",
  "memory_count": 150,
  "user_profile_count": 45,
  "group_profile_count": 3,
  "memory_breakdown": {
    "dm_memories": 50,
    "group_memories": 40,
    "user_memories": 35,
    "interaction_memories": 15,
    "cross_group_memories": 5,
    "conversation_summaries": 5
  }
}
```

---

### 4. User Profiles

#### `GET /v1/profiles/user/{universal_user_id}`
**Descripción**: Perfil global del usuario (agregado de todas sus interacciones).

**Cuándo usar**: Entender quién es el usuario antes de responder.

```bash
curl "http://136.118.160.81:8080/v1/profiles/user/telegram:88001?agent_id=jessexbt"
```

**Response**:
```json
{
  "profile_id": "uuid",
  "universal_user_id": "telegram:88001",
  "username": "elena_dev",
  "summary": "Full-stack developer with expertise in React, TypeScript...",
  "traits": {
    "engagement_level": {"value": "active", "confidence": 0.8}
  },
  "interests": [
    {"keyword": "TypeScript", "score": 0.9},
    {"keyword": "React", "score": 0.9}
  ],
  "expertise_level": {"value": "advanced", "confidence": 0.8},
  "entities": [
    {"type": "organization", "name": "Google", "context": "previous employer"}
  ],
  "total_messages_processed": 50
}
```

**Preguntas que responde**:
- "¿Quién es este usuario?" → Summary + traits
- "¿Es técnico o no-técnico?" → expertise_level
- "¿En qué está interesado?" → interests
- "¿Con qué empresas/proyectos está relacionado?" → entities

---

### 5. Group Profiles

#### `GET /v1/profiles/group/{group_id}`
**Descripción**: Perfil del grupo (tono, topics, usuarios activos).

```bash
curl "http://136.118.160.81:8080/v1/profiles/group/telegram_-100002?agent_id=jessexbt"
```

**Response**:
```json
{
  "group_id": "telegram_-100002",
  "group_name": "Dev Team",
  "summary": "Technical team discussing authentication and UI...",
  "tone": "professional",
  "topics": ["authentication", "UI design", "security"],
  "active_users": ["elena_dev", "carlos_audit", "maria_design"],
  "activity_level": "high"
}
```

**Preguntas que responde**:
- "¿De qué se habla en este grupo?" → topics
- "¿Quiénes son los más activos?" → active_users
- "¿Cómo debo hablar aquí?" → tone

---

#### `GET /v1/profiles/user/{user_id}/group/{group_id}`
**Descripción**: Perfil del usuario EN un grupo específico.

```bash
curl "http://136.118.160.81:8080/v1/profiles/user/telegram:88001/group/telegram_-100002?agent_id=jessexbt"
```

**Response**:
```json
{
  "user_id": "telegram:88001",
  "group_id": "telegram_-100002",
  "role_in_group": "developer",
  "topics_discussed": ["authentication", "React", "performance"],
  "interaction_style": "helpful",
  "message_count": 25
}
```

---

#### `GET /v1/profiles/group/{group_id}/members`
**Descripción**: Todos los miembros de un grupo con sus perfiles.

```bash
curl "http://136.118.160.81:8080/v1/profiles/group/telegram_-100002/members?agent_id=jessexbt"
```

---

### 6. User Facts (Evidence-Based)

#### `GET /v1/facts/{universal_user_id}`
**Descripción**: Hechos verificados sobre el usuario (con evidence count y confidence).

**Cuándo usar**: Cuando necesitas info confiable sobre el usuario.

```bash
curl "http://136.118.160.81:8080/v1/facts/telegram:88001?agent_id=jessexbt"
```

**Response**:
```json
{
  "user_id": "telegram:88001",
  "facts": [
    {
      "type": "expertise",
      "content": "Specializes in TypeScript and React development",
      "confidence": 0.9,
      "evidence_count": 3,
      "sources": ["dm_telegram:88001", "telegram_-100002"]
    },
    {
      "type": "personal",
      "content": "Based in Berlin, Germany",
      "confidence": 0.8,
      "evidence_count": 2,
      "sources": ["dm_telegram:88001"]
    }
  ],
  "total_facts": 5,
  "high_confidence_facts": 3
}
```

**Preguntas que responde**:
- "¿Qué sabemos con certeza sobre este usuario?" → facts con alta confidence
- "¿Dónde vive?" → facts type=personal
- "¿En qué es experto?" → facts type=expertise

---

#### `GET /v1/facts/{universal_user_id}/by-type`
**Descripción**: Filtrar facts por tipo.

```bash
curl "http://136.118.160.81:8080/v1/facts/telegram:88001/by-type?agent_id=jessexbt&fact_type=expertise"
```

**Fact Types**:
- `expertise` - Skills y conocimientos
- `preference` - Preferencias del usuario
- `personal` - Info personal (ubicación, trabajo)
- `interest` - Intereses
- `communication` - Estilo de comunicación

---

### 7. Summaries

#### `GET /v1/dm/{user_id}/summaries`
**Descripción**: Resúmenes de conversación DM (micro/chunk/block).

**Cuándo usar**: Contexto de conversaciones largas sin cargar todos los mensajes.

```bash
curl "http://136.118.160.81:8080/v1/dm/telegram:88001/summaries?agent_id=jessexbt"
```

**Response**:
```json
{
  "user_id": "telegram:88001",
  "summaries": {
    "micro": [
      {"summary": "Discussed TypeScript best practices...", "message_range": "10-19"}
    ],
    "chunk": [
      {"summary": "Week-long discussion about project architecture...", "message_range": "0-99"}
    ],
    "block": []
  }
}
```

**Thresholds**:
- Micro: cada 20 mensajes
- Chunk: cada 100 mensajes (5 micros)
- Block: cada 500 mensajes (5 chunks)

---

#### `GET /v1/groups/{group_id}/summaries`
**Descripción**: Resúmenes de grupo.

```bash
curl "http://136.118.160.81:8080/v1/groups/telegram_-100002/summaries?agent_id=jessexbt"
```

**Thresholds para grupos** (más altos por mayor volumen):
- Micro: cada 50 mensajes
- Chunk: cada 250 mensajes
- Block: cada 1250 mensajes

---

### 8. Spam Management

#### `GET /v1/spam/user/{user_id}/status`
**Descripción**: Ver si un usuario está bloqueado por spam.

```bash
curl "http://136.118.160.81:8080/v1/spam/user/telegram:88001/status?agent_id=jessexbt"
```

---

#### `POST /v1/spam/user/{user_id}/unblock`
**Descripción**: Desbloquear usuario.

```bash
curl -X POST "http://136.118.160.81:8080/v1/spam/user/telegram:88001/unblock?agent_id=jessexbt"
```

---

#### `GET /v1/spam/blocked`
**Descripción**: Listar todos los usuarios bloqueados.

```bash
curl "http://136.118.160.81:8080/v1/spam/blocked?agent_id=jessexbt"
```

---

### 9. Admin

#### `DELETE /v1/memory/reset/{agent_id}`
**Descripción**: Reset completo de un agente (Firestore + LanceDB).

```bash
curl -X DELETE "http://136.118.160.81:8080/v1/memory/reset/test_agent?confirm=true"
```

---

#### `POST /v1/memory/process-pending`
**Descripción**: Forzar procesamiento de mensajes pendientes.

```bash
curl -X POST "http://136.118.160.81:8080/v1/memory/process-pending?agent_id=jessexbt&group_id=telegram_-100002"
```

---

## Sistema de Ingestion (Detallado)

### Flujo Completo de un Mensaje

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POST /v1/memory/passive                              │
│                                                                              │
│  {                                                                           │
│    "agent_id": "jessexbt",                                                   │
│    "message": "I'm a Solidity developer",                                    │
│    "platform_identity": {                                                    │
│      "platform": "telegram",                                                 │
│      "telegramId": 123456,        ← ID único del usuario en la plataforma    │
│      "username": "alice_dev",      ← Username visible                        │
│      "chatId": "-100001"           ← Negativo = grupo, null = DM            │
│    },                                                                        │
│    "speaker": "alice_dev"          ← Quién dijo el mensaje                   │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 1: Determinar Contexto                                                 │
│                                                                              │
│  if chatId == null or chatId > 0:                                           │
│      effective_group_id = "dm_{user_id}"    ← Es un DM                      │
│  else:                                                                       │
│      effective_group_id = "telegram_{chatId}" ← Es un grupo                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 2: Generar Embedding para Spam Detection                               │
│                                                                              │
│  embedding = embedding_model.encode(message)  # 384 dims                     │
│  # Embedding se usa para:                                                    │
│  # 1. Detectar spam (similitud con mensajes recientes)                       │
│  # 2. Almacenar en Firestore para búsquedas rápidas                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 3: Spam Detection                                                      │
│                                                                              │
│  recent_embeddings = get_user_recent_messages(user_id, limit=5)              │
│                                                                              │
│  for each recent_msg in recent_embeddings:                                   │
│      similarity = cosine_similarity(new_embedding, recent_msg.embedding)     │
│      if similarity >= 0.92:                                                  │
│          is_spam = True                                                      │
│          reason = "high_similarity:0.95"                                     │
│          break                                                               │
│                                                                              │
│  if is_spam:                                                                 │
│      user.spam_score += 1.0                                                  │
│      if user.spam_score >= 3.0:                                             │
│          user.is_blocked = True  ← Usuario bloqueado                         │
│          return {is_blocked: true}                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 4: Guardar en Firestore Window                                         │
│                                                                              │
│  Collection: agents/{agent_id}/groups/{group_id}/recent_messages             │
│                                                                              │
│  Document:                                                                   │
│  {                                                                           │
│    content: "I'm a Solidity developer",                                      │
│    username: "alice_dev",                                                    │
│    platform_identity: {...},                                                 │
│    timestamp: "2026-01-30T...",                                              │
│    processed: false,           ← Marca para batch processing                 │
│    is_spam: false,                                                           │
│    spam_score: 0.0,                                                          │
│    embedding: [0.1, 0.2, ...]  ← Truncado a 384 dims                        │
│  }                                                                           │
│                                                                              │
│  Window Maintenance (smart cleanup):                                         │
│  1. Eliminar processed=true primero                                          │
│  2. Eliminar is_spam=true segundo                                            │
│  3. Eliminar más antiguos último                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PASO 5: Adaptive Threshold Check                                            │
│                                                                              │
│  threshold = calculate_adaptive_threshold(agent_id, group_id)                │
│  unprocessed = get_unprocessed_non_spam_messages(group_id)                   │
│                                                                              │
│  # Triggers para procesar:                                                   │
│  1. len(unprocessed) >= threshold        ← Suficientes mensajes             │
│  2. time_since_last_process >= 1 hora    ← Max wait time                    │
│  3. high_importance_msgs >= 3            ← Urgencia                         │
│                                                                              │
│  Threshold adaptativo:                                                       │
│  - Alta actividad (>20 msg/hr) → threshold menor (5-8)                      │
│  - Baja actividad (<2 msg/hr)  → threshold mayor (15-30)                    │
│  - Normal                      → threshold default (10)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                    threshold NOT met       threshold MET
                         │                       │
                         ▼                       ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────────┐
│  Return immediately:         │   │  PASO 6: Batch Processing                │
│  {                           │   │                                          │
│    success: true,            │   │  dialogues = convert_to_dialogues()      │
│    processing_scheduled: false│   │  memories = LLM.extract_memories()       │
│  }                           │   │                                          │
│                              │   │  # Paralelo (ThreadPoolExecutor):        │
│  Mensaje guardado en         │   │  ├─ Task: group_memories                 │
│  Firestore, esperando        │   │  ├─ Task: user_memories                  │
│  más mensajes.               │   │  ├─ Task: dm_memories (si es DM)         │
└─────────────────────────────┘   │  ├─ Task: user_profile                   │
                                   │  ├─ Task: group_profile (si >= 10 msgs) │
                                   │  ├─ Task: facts extraction              │
                                   │  └─ Task: summaries (si threshold)      │
                                   │                                          │
                                   │  mark_as_processed(doc_ids)              │
                                   │                                          │
                                   │  Return:                                 │
                                   │  {                                       │
                                   │    success: true,                        │
                                   │    processed: true,                      │
                                   │    memories_created: 5                   │
                                   │  }                                       │
                                   └─────────────────────────────────────────┘
```

---

### Parámetros de Ingestion

#### `POST /v1/memory/passive`

| Parámetro | Tipo | Requerido | Descripción |
|-----------|------|-----------|-------------|
| `agent_id` | string | ✅ | ID del agente (e.g., "jessexbt") |
| `message` | string | ✅ | Contenido del mensaje |
| `platform_identity` | object | ✅ | Identidad de la plataforma |
| `platform_identity.platform` | string | ✅ | "telegram", "xmtp", "farcaster" |
| `platform_identity.telegramId` | int | ✅* | ID único del usuario en Telegram |
| `platform_identity.username` | string | ⚪ | Username visible |
| `platform_identity.chatId` | string | ⚪ | ID del chat (negativo = grupo, null = DM) |
| `speaker` | string | ✅ | Quien envió el mensaje |

**Ejemplos de `chatId`:**
- `"-100001234567"` → Grupo de Telegram (negativo)
- `null` o ausente → DM privado
- `"123456789"` → Chat privado (positivo, tratado como DM)

#### Response

```json
{
  "success": true,
  "is_group": false,
  "group_id": "dm_telegram:123456",
  "user_id": "telegram:123456",
  "processing_scheduled": true,
  "memories_created": 0,
  "is_spam": false,
  "is_blocked": false,
  "spam_score": 0.0
}
```

---

### Sistema de Spam Detection

#### Cómo Funciona

```
┌─────────────────────────────────────────────────────────────────┐
│                    SPAM DETECTION FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Nuevo mensaje de User X                                         │
│           │                                                      │
│           ▼                                                      │
│  Obtener últimos 5 mensajes de User X en este grupo             │
│           │                                                      │
│           ▼                                                      │
│  Para cada mensaje reciente:                                     │
│      similarity = cosine(new_embedding, recent_embedding)        │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  REGLAS DE SPAM:                                    │        │
│  │                                                      │        │
│  │  1. similarity >= 0.92                               │        │
│  │     → SPAM (mensaje casi idéntico)                   │        │
│  │                                                      │        │
│  │  2. avg_similarity >= 0.87 AND count >= 2            │        │
│  │     → SPAM (patrón repetitivo)                       │        │
│  │                                                      │        │
│  │  3. user.spam_score >= 3.0                           │        │
│  │     → BLOCKED (usuario bloqueado)                    │        │
│  └─────────────────────────────────────────────────────┘        │
│           │                                                      │
│           ▼                                                      │
│  Si is_spam:                                                     │
│      user.spam_score += 1.0                                      │
│      mensaje.is_spam = true                                      │
│      (mensaje SE GUARDA pero NO se procesa en batch)            │
│           │                                                      │
│  Si NOT spam:                                                    │
│      user.spam_score -= 0.1 (decay gradual)                     │
│      mensaje.is_spam = false                                     │
│      (mensaje se procesa normalmente)                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuración de Spam

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `spam_similarity_threshold` | 0.92 | Similitud >= esto = spam |
| `spam_check_window` | 5 | Comparar con últimos N mensajes del usuario |
| `spam_score_decay` | 0.9 | Factor de decay por hora (score * 0.9^hours) |
| `spam_block_threshold` | 3.0 | Score >= esto = usuario bloqueado |

#### Endpoints de Spam

```bash
# Ver estado de spam de un usuario
GET /v1/spam/user/{user_id}/status?agent_id=jessexbt

# Response:
{
  "user_id": "telegram:123456",
  "spam_score": 1.5,
  "is_blocked": false,
  "block_threshold": 3.0,
  "total_spam_count": 2
}

# Desbloquear usuario
POST /v1/spam/user/{user_id}/unblock?agent_id=jessexbt

# Listar todos los bloqueados
GET /v1/spam/blocked?agent_id=jessexbt
```

---

### Adaptive Thresholds

El sistema ajusta dinámicamente cuántos mensajes esperar antes de procesar:

```
┌─────────────────────────────────────────────────────────────────┐
│                 ADAPTIVE THRESHOLD CALCULATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Actividad del grupo (msgs/hora)                                 │
│           │                                                      │
│           ▼                                                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  Alta actividad (>20 msg/hr):                        │        │
│  │      threshold = 5-8 mensajes                        │        │
│  │      window_size = 30-50 mensajes                    │        │
│  │      (Procesar más frecuente, más contexto)          │        │
│  │                                                      │        │
│  │  Actividad normal (2-20 msg/hr):                     │        │
│  │      threshold = 10 mensajes (default)               │        │
│  │      window_size = 15 mensajes                       │        │
│  │                                                      │        │
│  │  Baja actividad (<2 msg/hr):                         │        │
│  │      threshold = 15-30 mensajes                      │        │
│  │      window_size = 10-12 mensajes                    │        │
│  │      (Esperar más contexto, menos storage)           │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
│  TRIGGERS ADICIONALES:                                           │
│  ├─ Max wait time: 1 hora → procesar aunque no haya threshold   │
│  └─ Urgency: 3+ mensajes importantes → procesar inmediato       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuración de Thresholds

| Parámetro | Valor Default | Descripción |
|-----------|---------------|-------------|
| `min_batch_size` | 5 | Mínimo de mensajes para procesar |
| `max_batch_size` | 30 | Máximo de mensajes por batch |
| `default_batch_size` | 10 | Threshold default |
| `high_activity_threshold` | 20.0 | msgs/hr = alta actividad |
| `low_activity_threshold` | 2.0 | msgs/hr = baja actividad |
| `max_wait_time_seconds` | 3600 | 1 hora max sin procesar |
| `min_window_size` | 10 | Mínimo mensajes en window |
| `max_window_size` | 50 | Máximo mensajes en window |

---

### Batch Processing (Qué se Genera)

Cuando se dispara el batch processing, se ejecutan en paralelo:

| Task | Condición | Output |
|------|-----------|--------|
| **DM Memories** | Es DM | `dm_memories` table con `is_shareable` flag |
| **Group Memories** | Es Grupo | `group_memories` table (conocimiento compartido) |
| **User Memories** | Es Grupo | `user_memories` table (lo que cada user dijo) |
| **Interaction Memories** | Es Grupo | `interaction_memories` table (user-to-user) |
| **User Profile** | Siempre | `user_profiles` table (perfil global del user) |
| **Group Profile** | >= 10 msgs en grupo | `group_profiles` table |
| **User Facts** | Siempre | `user_facts` table (facts con evidence) |
| **DM Summaries** | >= 20 msgs DM | `dm_summaries` table (micro) |
| **Group Summaries** | >= 50 msgs grupo | `group_summaries` table (micro) |

---

## Flujo Típico de Integración

### 1. Usuario envía mensaje en grupo
```bash
# 1. Ingesta pasiva
POST /v1/memory/passive
{
  "agent_id": "jessexbt",
  "message": "Can someone help with smart contract security?",
  "platform_identity": {"platform": "telegram", "telegramId": 123, "chatId": "-100001"},
  "speaker": "alice"
}

# 2. Obtener contexto para responder
POST /v1/memory/context
{
  "agent_id": "jessexbt",
  "query": "smart contract security help",
  "platform_identity": {"platform": "telegram", "telegramId": 123, "chatId": "-100001"},
  "involved_users": ["telegram:123"]
}

# 3. El agente responde usando formatted_context
```

### 2. Usuario envía mensaje en DM
```bash
# 1. Ingesta pasiva (sin chatId = DM)
POST /v1/memory/passive
{
  "agent_id": "jessexbt",
  "message": "I'm a Solidity auditor based in Singapore",
  "platform_identity": {"platform": "telegram", "telegramId": 123, "username": "alice"},
  "speaker": "alice"
}
# → Genera dm_memories con is_shareable=true (info profesional)

# 2. Más tarde, en un grupo, el agente puede ver esta info
# cuando alice habla (Speaker's Personal Context)
```

---

## Tablas LanceDB

| Tabla | Scope | Contenido |
|-------|-------|-----------|
| `memories` | Per-agent | DM memories (MemoryEntry) con `is_shareable` |
| `group_memories` | Per-agent | Conocimiento del grupo |
| `user_memories` | Per-agent | Lo que cada user dijo en grupos |
| `interaction_memories` | Per-agent | Interacciones user-to-user |
| `cross_group_memories` | Per-agent | Patterns across groups |
| `user_profiles` | Global | Perfil global de usuarios (LLM-generated) |
| `group_profiles` | Global | Perfil de grupos (LLM-generated) |
| `user_facts` | Global | Facts verificados de usuarios |
| `dm_summaries` | Global | Resúmenes de DMs (micro/chunk/block) |
| `group_summaries` | Global | Resúmenes de grupos (micro/chunk/block) |

---

## Estado Actual de Generación

| Feature | Estado | Descripción |
|---------|--------|-------------|
| **DM Memories** | ✅ | Con `is_shareable` flag |
| **Group Memories** | ✅ | group/user/interaction memories |
| **User Profiles** | ✅ | LLM-generated, actualizado cada batch |
| **Group Profiles** | ✅ | LLM-generated cuando >= 10 msgs en grupo |
| **User Facts** | ✅ | Evidence-based extraction |
| **DM Summaries** | ✅ | Micros generados inline. Chunks/blocks via job service |
| **Group Summaries** | ✅ | Micros generados inline. Chunks/blocks via job service |
| **Cross-Group Memories** | ⚠️ | Requiere patterns en 2+ grupos/DMs |

---

## Arquitectura de Summarization

### Dos Servicios

```
┌─────────────────────────────────────────────────────────────────┐
│  a0x-memory (API principal)                                      │
│                                                                  │
│  Durante ingestion:                                              │
│  - Genera MICROS cuando hay suficientes mensajes                │
│    - DM: cada 20 mensajes                                        │
│    - Group: cada 50 mensajes                                     │
│                                                                  │
│  NO genera chunks/blocks (evita sobrecarga durante ingestion)    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Cloud Scheduler (periódico)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  a0x-memory-jobs (Job service)                                   │
│                                                                  │
│  POST /jobs/consolidate                                          │
│                                                                  │
│  Agrega summaries:                                               │
│  - 5 micros  → 1 chunk  (luego elimina micros)                  │
│  - 5 chunks  → 1 block  (luego elimina chunks)                  │
│  - 5 blocks  → 1 era    (luego elimina blocks)                  │
│                                                                  │
│  Esto mantiene el storage bounded mientras preserva historia.    │
└─────────────────────────────────────────────────────────────────┘
```

### Endpoints del Job Service (a0x-memory-jobs)

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/jobs/consolidate` | POST | Full consolidation de todos los grupos y DMs |
| `/jobs/consolidate/{context_id}` | POST | Consolidar grupo o DM específico |
| `/jobs/cleanup` | POST | Limpiar summaries huérfanos |
| `/jobs/stats` | GET | Estadísticas del store |
| `/health` | GET | Health check |

### Flujo de Summarization

```
Mensajes nuevos
      │
      ▼
┌─────────────────┐
│  Ingestion      │
│  (a0x-memory)   │
│                 │
│  if msgs >= 20  │──────────► Crear MICRO (DM)
│  if msgs >= 50  │──────────► Crear MICRO (Group)
└─────────────────┘
      │
      │ (micros acumulados)
      ▼
┌─────────────────┐
│  Consolidation  │     Cloud Scheduler
│  (jobs service) │◄────(cada X horas)
│                 │
│  5 micros       │──────────► Crear CHUNK, eliminar micros
│  5 chunks       │──────────► Crear BLOCK, eliminar chunks
│  5 blocks       │──────────► Crear ERA, eliminar blocks
└─────────────────┘
```

### Thresholds de Summarization

| Contexto | Nivel | Trigger | Mensajes Cubiertos |
|----------|-------|---------|-------------------|
| **DM** | Micro | 20 msgs | ~20 |
| **DM** | Chunk | 5 micros | ~100 |
| **DM** | Block | 5 chunks | ~500 |
| **DM** | Era | 5 blocks | ~2500 |
| **Group** | Micro | 50 msgs | ~50 |
| **Group** | Chunk | 5 micros | ~250 |
| **Group** | Block | 5 chunks | ~1250 |
| **Group** | Era | 5 blocks | ~6250 |

### Estado Actual - Summarization

| Nivel | Generación | Estado |
|-------|------------|--------|
| **Micro** | Inline (durante ingestion) | ✅ Implementado |
| **Chunk** | Job service | ✅ Implementado (requiere deploy del job) |
| **Block** | Job service | ✅ Implementado (requiere deploy del job) |
| **Era** | Job service | ✅ Implementado (requiere deploy del job) |

---

## Jobs Service - Arquitectura Completa

### Distribución de Tareas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  a0x-memory (API principal) - INLINE                                         │
│                                                                              │
│  Durante ingestion (crítico para respuesta):                                 │
│  ✅ Spam detection                                                           │
│  ✅ Guardar en Firestore window                                              │
│  ✅ Memory extraction (dm/group/user/interaction memories)                   │
│  ✅ Micro summary generation                                                 │
│  ✅ Conversation summary update                                              │
│                                                                              │
│  ~2-3 LLM calls por batch (optimizado)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                              │
                              │ Cloud Scheduler (periódico)
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  a0x-memory-jobs (Job service) - PERIÓDICO                                   │
│                                                                              │
│  Jobs de consolidación:                                                      │
│  ✅ Summary aggregation (micros → chunks → blocks → eras)                   │
│  🔜 Profile generation (user, group, user-in-group)                         │
│  🔜 Fact extraction & consolidation                                          │
│  🔜 Cross-group memory consolidation                                         │
│  🔜 Decay updates & cleanup                                                  │
│                                                                              │
│  Ejecuta trabajo pesado sin bloquear ingestion                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Jobs Planificados

| Job | Endpoint | Frecuencia Sugerida | Descripción |
|-----|----------|---------------------|-------------|
| **Consolidation** | `POST /jobs/consolidate` | Cada 1h | micros → chunks → blocks → eras |
| **Profiles** | `POST /jobs/profiles` | Cada 2h | Regenerar user/group profiles |
| **Facts** | `POST /jobs/facts` | Cada 2h | Extraer y consolidar facts |
| **Cross-group** | `POST /jobs/cross-group` | Cada 6h | Detectar patterns cross-context |
| **Maintenance** | `POST /jobs/maintenance` | Cada 24h | Decay updates + cleanup |

### Beneficios de esta Arquitectura

1. **Ingestion más rápida**: Solo 2-3 LLM calls vs 7+ anteriormente
2. **Escalabilidad**: Jobs pueden correr en instancias separadas
3. **Resiliencia**: Si un job falla, no afecta la ingestion
4. **Costos optimizados**: Jobs pueden usar modelos más lentos/baratos
5. **Mantenibilidad**: Lógica separada, más fácil de debuggear

**Nota:** El job service (`a0x-memory-jobs`) debe desplegarse y configurarse con Cloud Scheduler para que los jobs se ejecuten periódicamente.

**Nota**: User Profiles y Group Profiles NO son summaries jerárquicos. Son análisis LLM que se regeneran/actualizan con cada batch de mensajes procesados.

---

## Arquitectura de Context Retrieval

### Dos Capas de Contexto

El sistema combina dos fuentes de contexto complementarias:

```
┌─────────────────────────────────────────────────────────────┐
│  1. FIRESTORE WINDOW (siempre incluido)                     │
│     → Últimos 10-50 mensajes literales de la conversación   │
│     → Contexto inmediato, sin procesar                      │
│     → NO requiere query - siempre se incluye                │
└─────────────────────────────────────────────────────────────┘
                           +
┌─────────────────────────────────────────────────────────────┐
│  2. LANCEDB MEMORIES (búsqueda semántica con planning)      │
│     → Conocimiento extraído y consolidado                   │
│     → Búsqueda basada en la query del usuario               │
│     → Usa planning para generar múltiples sub-queries       │
└─────────────────────────────────────────────────────────────┘
```

**¿Por qué dos capas?**
- **Window**: "¿Qué se dijo AHORA?" - Contexto inmediato de la conversación actual
- **Memories**: "¿Qué SABEMOS relevante?" - Conocimiento acumulado relevante a la pregunta

---

### Sistema de Planning

Cuando se hace una búsqueda de contexto, el sistema NO busca directamente con la query. Usa un **planner LLM** que:

1. **Analiza la query** - Identifica tipo de pregunta, entidades, información requerida
2. **Genera sub-queries** - Crea 1-3 queries optimizadas para buscar diferentes aspectos
3. **Fan-out search** - Ejecuta todas las queries en paralelo en todas las tablas
4. **Merge + Dedupe** - Combina resultados y elimina duplicados
5. **Rerank** - Ordena por relevancia combinada

```
Query: "What does Elena know about smart contracts?"
                    ↓
┌─────────────────────────────────────────────────────────────┐
│                    PLANNING (1 LLM call)                    │
│                                                             │
│  question_type: "factual"                                   │
│  key_entities: ["Elena", "smart contracts"]                 │
│  required_info: ["Elena's blockchain expertise",            │
│                  "smart contract projects"]                 │
│                                                             │
│  Generated queries:                                         │
│  1. "What does Elena know about smart contracts?"           │
│  2. "Elena's blockchain and Solidity experience"            │
│  3. "Elena's Web3 projects"                                 │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│              FAN-OUT SEARCH (paralelo)                      │
│                                                             │
│  Query 1 → group_memories, user_memories, dm_memories       │
│  Query 2 → group_memories, user_memories, dm_memories       │
│  Query 3 → group_memories, user_memories, dm_memories       │
└─────────────────────────────────────────────────────────────┘
                    ↓
            Merge + Dedupe + Rerank
                    ↓
              Resultados finales
```

---

### Límites de Summaries en Context

Cuando se incluyen summaries jerárquicos en el contexto, se aplican límites:

| Nivel | Límite Default | Cobertura Aproximada |
|-------|----------------|----------------------|
| **Block** | 2 más recientes | ~2500 mensajes históricos |
| **Chunk** | 3 más recientes | ~750 mensajes recientes |
| **Micro** | 5 más recientes | ~250 mensajes actuales |

**Formato en contexto:**
```
[Historical] Messages 0-1250: Discussion about authentication... (Topics: auth, security)
[Recent period] Messages 1250-1500: Team worked on UI redesign...
[Latest activity] msgs 1450-1500: UI | msgs 1400-1450: testing | msgs 1350-1400: deployment
```

---

### Tablas Buscadas por Contexto

| Contexto | Tablas Buscadas |
|----------|-----------------|
| **DM** | dm_memories, cross_group_memories, user_facts, user_profiles |
| **Group** | group_memories, user_memories, interaction_memories, speaker's dm_memories (shareable), user_facts, group_summaries |

---

## Modelo de Privacidad Detallado

### `is_shareable` Flag

Las DM memories tienen un flag `is_shareable` decidido por LLM:

| is_shareable | Ejemplo | Visible en Grupo |
|--------------|---------|------------------|
| `true` | "Soy desarrollador Solidity con 5 años" | ✅ Sí (cuando el user habla) |
| `false` | "Tengo problemas financieros" | ❌ No |

**Criterios para `is_shareable=true`:**
- Información profesional (skills, experiencia, proyectos)
- Preferencias técnicas públicas
- Datos de contacto profesional

**Criterios para `is_shareable=false`:**
- Información personal sensible
- Problemas o quejas privadas
- Contexto específico de la conversación DM

### Visibilidad por Contexto

```
┌─────────────────────────────────────────────────────────────┐
│  EN DM CON USER X                                           │
│                                                             │
│  ✅ VE:                                                     │
│     - Historial de DMs con X                                │
│     - Todo lo que X dijo en grupos (público)                │
│     - Facts de X                                            │
│     - Cross-group memories de X                             │
│                                                             │
│  ❌ NO VE:                                                  │
│     - DMs de otros usuarios                                 │
│     - Lo que OTROS dijeron en grupos                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  EN GRUPO (Speaker = X)                                     │
│                                                             │
│  ✅ VE:                                                     │
│     - Historial del grupo                                   │
│     - Memorias del grupo                                    │
│     - DM memories de X con is_shareable=true                │
│     - Facts de X                                            │
│     - Profiles de usuarios mencionados                      │
│                                                             │
│  ❌ NO VE:                                                  │
│     - DMs de otros usuarios                                 │
│     - Memorias de otros grupos                              │
│     - DM memories de X con is_shareable=false               │
└─────────────────────────────────────────────────────────────┘
```

**Importante:** En grupo, SOLO el speaker actual ve sus propias DM memories shareable. Si Carlos habla, NO ve las DM memories de Elena.
