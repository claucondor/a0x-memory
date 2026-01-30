# Arquitectura Híbrida de User Profiles para a0x-memory

## 🎯 Objetivo

Sistema de user profiles que se ejecuta en **background**, sin bloquear las respuestas del agente, usando **solo a0x-models** (sin OpenRouter por ahora).

---

## 📊 Resultados del Test (Referencia)

| Métrica | Valor | Nota |
|---------|-------|------|
| **Tiempo total (15 msgs, paralelo)** | ~28s | Aceptable para background |
| **Speedup paralelización** | 2.2x | **Crítico** - debe ser paralelo |
| **Per message avg** | ~1.9s | Más eficiente que batch pequeño |
| **Bottleneck principal** | Clasificaciones (87%) | mDeBERTa-v3 es lento |

---

## 🏗️ Arquitectura Propuesta

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          AGENT RESPONSE FLOW                          │
│                        (Bloqueante, crítico)                          │
└─────────────────────────────────────────────────────────────────────────┘
     ↓
     ┌──────────────────────────────────────────────────────┐
     │  1. Agent recibe mensaje                             │
     │  2. Recupera profile CACHED (instantáneo)           │
     │  3. Genera respuesta usando profile existente       │
     │  4. Retorna respuesta al usuario                     │
     └──────────────────────────────────────────────────────┘
     ↓ (no bloquea)
┌─────────────────────────────────────────────────────────────────────────┐
│                      PROFILE UPDATE FLOW                             │
│                      (Background, no bloqueante)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Componentes

### 1. **UserProfileStore** (LanceDB)

```python
# Tabla: user_profiles
{
    "user_id": str,              # PK

    # Platform identifiers
    "telegram_id": str | None,
    "telegram_handle": str | None,
    "twitter_id": str | None,
    "wallet_address": str | None,

    # Profile data
    "summary": str,              # 2-3 sentence bio
    "summary_updated_at": str,

    # Structured traits con CONFIDENCE
    "traits": {
        "expertise_level": {"value": "advanced", "confidence": 0.8},
        "communication_style": {"value": "technical", "confidence": 0.7},
        "domains": {"value": ["trading", "defi"], "confidence": 0.75}
    },

    # Extracted data
    "interests": List[dict],     # Top 10 keywords con scores
    "entities": {
        "persons": List[str],
        "organizations": List[str],
        "locations": List[str]
    },

    # Metadata
    "message_count_last_update": int,
    "last_profile_update": str,
    "profile_version": int,
    "first_seen": str,
    "last_active": str
}
```

### 2. **UserProfileExtractor** (a0x-models API)

```python
class UserProfileExtractor:
    """
    Extrae perfil usando a0x-models API
    - Clasificaciones en PARALELO (2.2x speedup)
    - Todas las operaciones no bloqueantes
    """

    def extract_profile(
        self,
        messages: List[str],
        existing_profile: dict | None = None
    ) -> dict:
        """
        Extrae perfil completo en ~28s (15 mensajes)

        Returns:
            UserProfile completo
        """
```

### 3. **UserProfileService** (Orquestador)

```python
class UserProfileService:
    """
    Maneja actualizaciones de perfil en background
    """

    def __init__(self, extractor: UserProfileExtractor, store: UserProfileStore):
        self.extractor = extractor
        self.store = store
        self.update_queue = {}  # user_id -> messages_buffer

    # ===== CORE METHODS =====

    async def get_profile(self, user_id: str) -> dict:
        """Obtener perfil CACHED (instantáneo)"""
        return await self.store.get(user_id)

    async def add_messages(self, user_id: str, messages: List[str]):
        """Agregar mensajes al buffer (non-blocking)"""
        if user_id not in self.update_queue:
            self.update_queue[user_id] = []
        self.update_queue[user_id].extend(messages)

        # Trigger update si hay suficientes mensajes
        if len(self.update_queue[user_id]) >= 10:
            await self._schedule_update(user_id)

    async def _schedule_update(self, user_id: str):
        """Programar actualización en background (fire-and-forget)"""
        # No await! Se ejecuta en background
        self._update_profileInBackground(user_id)

    def _update_profileInBackground(self, user_id: str):
        """Actualiza perfil en background thread"""
        messages = self.update_queue.get(user_id, [])
        if not messages:
            return

        # Extraer perfil (toma ~28s)
        profile = self.extractor.extract_profile(messages)

        # Guardar en store
        self.store.save(user_id, profile)

        # Limpiar buffer
        self.update_queue[user_id] = []
```

### 4. **UpdateTrigger** (Cron Job)

```python
class UpdateTrigger:
    """
    Job cron que ejecuta actualizaciones pendientes
    - Corre cada 5 minutos
    - Procesa usuarios con mensajes acumulados
    """

    async def process_pending_updates(self):
        """Procesa todos los usuarios con mensajes pendientes"""
        for user_id, messages in self.service.update_queue.items():
            if len(messages) >= 5:  # Threshold mínimo
                await self.service._update_profileInBackground(user_id)
```

---

## 🔄 Flujo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCENARIO: Nuevo mensaje                      │
└─────────────────────────────────────────────────────────────────┘

Usuario envía mensaje
    ↓
┌─────────────────────────────────────────────────────────────────┐
│  AGENT (Síncrono, debe ser rápido)                              │
│  1. add_messages(user_id, [mensaje])                            │
│     → Agrega al buffer, NO bloquea                               │
│  2. get_profile(user_id)                                        │
│     → Retorna perfil CACHED inmediatamente                      │
│  3. Genera respuesta usando profile actualizado                 │
│  4. Responde al usuario                                         │
└─────────────────────────────────────────────────────────────────┘
    ↓ (en paralelo, background)
┌─────────────────────────────────────────────────────────────────┐
│  PROFILE UPDATE (Background, no bloquea al agente)               │
│  1. Verificar: ¿Hay 10+ mensajes en buffer?                    │
│  2. SÍ → Extraer perfil (~28s, paralelo)                        │
│  3. Guardar nuevo perfil en LanceDB                            │
│  4. Limpiar buffer                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📋 Política de Actualización

### ¿Cuándo actualizar?

| Trigger | Acción | Razón |
|---------|--------|-------|
| **10 mensajes acumulados** | Extraer perfil | Suficiente data para cambios |
| **24 horas desde último update** | Extraer aunque sean <10 | Mantener fresco |
| **Usuario explícitamente lo pide** | Extraer inmediatamente | Comando `/profile refresh` |

### ¿Cuándo NO actualizar?

| Situación | Acción |
|-----------|--------|
| Menos de 5 mensajes | Acumular en buffer |
| Último update hace <1 hora | Usar caché |
| Agente ocupado respondiendo | No disparar update |

---

## 🚀 Optimizaciones Implementadas

### 1. **Clasificaciones Paralelas**
```python
# Antes: 50s (secuencial)
expertise = classify(labels_1)  # 12s
style = classify(labels_2)      # 12s
domains = classify(labels_3)    # 26s

# Después: 24s (paralelo)
with ThreadPoolExecutor(max_workers=3) as executor:
    expertise, style, domains = executor.map(classify, [...])
# Speedup: 2.2x
```

### 2. **Batch Processing**
- 1 mensaje: 16s por mensaje
- 15 mensajes: 1.9s por mensaje
- **Eficiencia: 8.4x mejor en batch**

### 3. **Background Execution**
- Agente no espera por profile update
- Respuesta inmediata usando caché
- Update se completa después

---

## 💾 Storage Strategy

### Opción A: Todo en LanceDB (RECOMENDADO)

```python
# Ventajas:
- Single deployment
- Mismo stack que memories
- Fácil de exportar/backup

# Tablas:
- memories (existente)
- user_profiles (nueva)
```

### Opción B: LanceDB + Firestore

```python
# Ventajas:
- Queries exactas más rápidas (telegram_id lookup)
- Mejor para multi-tenant

# Trade-off:
- Más complejo
- Dos sistemas que mantener
```

**Recomendación:** Empezar con Opción A, migrar si escala.

---

## 🔍 "Cambia Significativamente" - Aclaración

El usuario preguntó qué significa esto. Aquí está la aclaración:

### No Usamos "Detección de Cambio Significativo"

En su lugar usamos **thresholds simples basados en cantidad**:

```python
# ✅ LO QUE HACEMOS (Simple y determinista):
if new_messages_count >= 10:
    update_profile()

# ❌ LO QUE NO HACEMOS (Complejo e innecesario):
similarity = cosine_distance(old_profile, new_profile)
if similarity < 0.7:  # "cambió significativamente"
    update_profile()
```

**¿Por qué?**
1. Más simple de implementar
2. Predecible y testeable
3. Los cambios son inevitables con N mensajes
4. El perfil evoluciona naturalmente

---

## 📐 Esquema de Base de Datos

```python
# LanceDB schema
user_profiles = pa.schema([
    # Primary key
    pa.field("user_id", pa.string()),

    # Platform identifiers (todos nullable)
    pa.field("telegram_id", pa.string()),
    pa.field("telegram_handle", pa.string()),
    pa.field("twitter_id", pa.string()),
    pa.field("wallet_address", pa.string()),

    # Profile data
    pa.field("summary", pa.string()),
    pa.field("summary_updated_at", pa.string()),

    # Structured traits (JSONB para flexibilidad)
    pa.field("traits", pa.string()),  # JSON string
    pa.field("interests", pa.list_(pa.string())),

    # Entities (arrays)
    pa.field("persons", pa.list_(pa.string())),
    pa.field("organizations", pa.list_(pa.string())),
    pa.field("locations", pa.list_(pa.string())),

    # Metadata
    pa.field("message_count_last_update", pa.int64()),
    pa.field("last_profile_update", pa.string()),
    pa.field("profile_version", pa.int64()),
    pa.field("first_seen", pa.string()),
    pa.field("last_active", pa.string()),

    # Index para búsquedas rápidas
    pa.field("summary_vector", pa.list_(pa.float32())),  # Para semantic search de profiles
])
```

---

## 🎬 Diagrama de Secuencia

```
Usuario        Agent          ProfileService      Extractor       LanceDB
  │              │                  │               │               │
  │  mensaje     │                  │               │               │
  ├─────────────>│                  │               │               │
  │              │                  │               │               │
  │              │ add_messages()   │               │               │
  │              ├─────────────────>│               │               │
  │              │                  │ [buffer += 1] │               │
  │              │<─────────────────┤               │               │
  │              │                  │               │               │
  │              │ get_profile()    │               │               │
  │              ├─────────────────>│               │               │
  │              │                  │               │               │
  │              │                  ├───────────────>│               │
  │              │                  │               │               │
  │              │                  │<───────────────┤ (cached)      │
  │              │<─────────────────┤               │               │
  │              │                  │               │               │
  │  respuesta   │                  │               │               │
  │<─────────────┤                  │               │               │
  │              │                  │               │               │
  │              │                  │ (background)  │               │
  │              │                  ├───────────────>│               │
  │              │                  │  extract()    │               │
  │              │                  │  (~28s)       │               │
  │              │                  │<───────────────┤               │
  │              │                  │               │               │
  │              │                  ├───────────────────────────────>│
  │              │                  │  save()       │               │
  │              │                  │<───────────────────────────────┤
  │              │                  │               │               │
```

---

## ✅ Plan de Implementación

### Phase 1: Storage Layer (1 día)
- [ ] Crear tabla `user_profiles` en LanceDB
- [ ] Implementar `UserProfileStore` class
- [ ] Métodos: get, save, delete, list_by_platform

### Phase 2: Extraction Layer (2 días)
- [ ] Implementar `UserProfileExtractor` con a0x-models
- [ ] Clasificaciones paralelas (ThreadPoolExecutor)
- [ ] Tests completos

### Phase 3: Service Layer (2 días)
- [ ] `UserProfileService` con buffer
- [ ] Background updates (fire-and-forget)
- [ ] Update trigger (cron)

### Phase 4: Integration (1 día)
- [ ] Integrar con agent execution
- [ ] Inyectar profile en system prompt
- [ ] Commands: `/profile`, `/profile refresh`

---

## 📝 Resumen Ejecutivo

| Aspecto | Decisión |
|---------|----------|
| **Storage** | LanceDB (todo junto) |
| **Update trigger** | Cada 10 mensajes o 24h |
| **Execution** | Background, no bloqueante |
| **Optimización** | Clasificaciones paralelas (2.2x) |
| **Cache** | Perfil siempre en memoria |
| **OpenRouter** | Pendiente (test después) |

---

## 🚀 Próximos Pasos

1. **Revisar esta arquitectura** - ¿Te gusta?
2. **Decidir storage** - ¿LanceDB o LanceDB+Firestore?
3. **Empezar implementación** - Phase 1 (Storage)
