# Group Memory & User Identity Design

Diseño para manejo de memorias en grupos de Telegram y sistema de identidad cross-platform.

---

## Problema

1. Agentes operan en múltiples contextos: DMs, grupos de Telegram, Twitter público
2. Un usuario puede interactuar con el agente en varios contextos
3. Privacidad: lo que se dice en un grupo no debe filtrarse a DMs sin consentimiento
4. Identidad: un usuario puede tener múltiples handles (telegram, twitter, wallet, etc.)

---

## Arquitectura de Capas

```
┌─────────────────────────────────────────────────────────┐
│                    User Profiles                         │
│         (identidad unificada cross-platform)            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                      Memories                            │
│    (con privacy_scope y context metadata)               │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   Group Contexts                         │
│         (metadata del grupo: tema, reglas)              │
└─────────────────────────────────────────────────────────┘
```

---

## Schema: User Profiles

Tabla separada para identidad de usuarios.

```python
user_profiles = {
    "user_id": str,              # UUID interno (PK)

    # Identifiers (cualquiera puede ser null)
    "telegram_handle": str,      # "@alice_tg"
    "telegram_id": str,          # "123456789" (numeric ID, más estable)
    "twitter_handle": str,       # "@alice"
    "twitter_id": str,           # numeric ID
    "farcaster_handle": str,     # "@alice.eth"
    "farcaster_fid": str,        # Farcaster ID
    "wallet_address": str,       # "0x123..." (primary wallet)
    "wallets": List[str],        # ["0x123...", "0x456..."] (all known)
    "basename": str,             # "alice.base"
    "ens": str,                  # "alice.eth"

    # Profile summary (LLM-generated, modelo TBD)
    "summary": str,              # "Interested in DeFi, works in crypto..."
    "summary_updated_at": str,   # ISO timestamp

    # Metadata
    "first_seen": str,           # ISO timestamp
    "last_active": str,          # ISO timestamp
    "created_by_agent": str,     # which agent first saw this user

    # Privacy settings (user-controlled)
    "cross_context_enabled": bool,  # default: False
    "data_retention_days": int,     # null = forever, or 30, 90, etc.
}
```

**Índices:**
- Primary: `user_id`
- Unique: `telegram_id`, `twitter_id`, `farcaster_fid` (si no null)
- Search: `wallet_address`, `telegram_handle`, `twitter_handle`

---

## Schema: Memories (Updated)

Extensión del schema actual para soportar grupos y privacidad.

```python
memories = {
    # Tenant (existing)
    "agent_id": str,             # which agent
    "user_id": str,              # who said it / about whom

    # NEW: Group context
    "group_id": str | None,      # null = DM or public timeline
    "group_platform": str | None, # "telegram" | "discord" | etc.

    # NEW: Context classification
    "context_type": str,         # "dm" | "group" | "public_timeline" | "broadcast"

    # NEW: Privacy scope
    "privacy_scope": str,        # "private" | "group_only" | "cross_context"

    # Existing fields
    "entry_id": str,
    "lossless_restatement": str,
    "keywords": List[str],
    "timestamp": str,
    "location": str,
    "persons": List[str],
    "entities": List[str],
    "topic": str,
    "vector": List[float],

    # NEW: Additional metadata
    "platform": str,             # "telegram" | "twitter" | "farcaster"
    "message_id": str | None,    # original message ID for dedup
    "reply_to_message_id": str | None,  # if this was a reply
}
```

**Privacy scope rules:**
| context_type | default privacy_scope | can be changed to |
|--------------|----------------------|-------------------|
| dm | private | cross_context (user opt-in) |
| group | group_only | cross_context (user opt-in) |
| public_timeline | cross_context | private (user request) |
| broadcast | cross_context | N/A |

---

## Schema: Group Contexts

Metadata sobre grupos donde el agente participa.

```python
group_contexts = {
    "group_id": str,             # PK (platform-specific ID)
    "platform": str,             # "telegram" | "discord"
    "agent_id": str,             # which agent is in this group

    # Group info
    "name": str,                 # "Crypto Traders"
    "description": str | None,   # group description if available
    "member_count": int | None,  # approximate

    # LLM-generated understanding (modelo TBD)
    "topic_summary": str,        # "Group discusses crypto trading and market analysis"
    "tone": str,                 # "casual" | "professional" | "memey"
    "language": str,             # "en" | "es" | "mixed"
    "summary_updated_at": str,

    # Agent behavior config
    "respond_only_when_mentioned": bool,  # default: True
    "learn_from_all_messages": bool,      # default: True
    "memory_retention_days": int | None,  # null = forever

    # Metadata
    "joined_at": str,
    "last_activity": str,
    "is_active": bool,           # agent still in group?
}
```

---

## Query Patterns

### 1. Agent responding in DM
```python
# Get user's DM memories + cross_context memories
memories.filter(
    agent_id=agent,
    user_id=user,
    OR(
        privacy_scope="private" AND group_id=None,
        privacy_scope="cross_context"
    )
)
```

### 2. Agent responding in Group
```python
# Get group memories + user's cross_context + group context
memories.filter(
    agent_id=agent,
    OR(
        group_id=group,  # all group memories
        user_id=user AND privacy_scope="cross_context"  # user's public knowledge
    )
)
```

### 3. Building user summary
```python
# Only cross_context memories for global summary
memories.filter(
    agent_id=agent,
    user_id=user,
    privacy_scope="cross_context"
)
```

### 4. "What does agent know about me?" (user request)
```python
# All memories about user (for transparency)
memories.filter(
    agent_id=agent,
    user_id=user
)
# Return grouped by context with privacy_scope visible
```

---

## Identity Linking

### Problema
Un usuario puede ser:
- `@alice` en Twitter
- `@alice_tg` en Telegram
- `0x123...` wallet
- `alice.base` basename

¿Cómo sabemos que son la misma persona?

### Métodos de Linking

| Method | Confidence | Implementation |
|--------|------------|----------------|
| User declares | High | "my twitter is @alice" → link |
| Wallet signature | Very High | Sign message proving ownership |
| Same display name | Low | Heuristic, requires confirmation |
| Shared in bio | Medium | Parse Twitter bio for wallet/basename |
| On-chain data | High | Basename → wallet lookup |

### Linking Flow
```
1. User interacts on Telegram as @alice_tg
2. Agent creates user_profile with telegram_id only
3. User says "my wallet is 0x123"
4. Agent asks for confirmation / signature (optional)
5. Agent updates user_profile.wallet_address = "0x123"
6. If 0x123 has basename "alice.base", add that too
7. Now agent can recognize user across platforms
```

### Conflicto de Merge
Si dos user_profiles need to merge:
```
Profile A: telegram_id=123, wallet=null
Profile B: telegram_id=null, wallet=0x123

User proves they own both → MERGE
- Keep older user_id as primary
- Combine all identifiers
- Combine all memories (update user_id FK)
```

---

## Privacy Implementation

### Default Behavior
```python
def get_default_privacy_scope(context_type: str) -> str:
    if context_type == "dm":
        return "private"
    elif context_type == "group":
        return "group_only"
    elif context_type == "public_timeline":
        return "cross_context"
    else:
        return "private"  # safe default
```

### User Commands
```
/privacy status     → Show current settings
/privacy share      → Enable cross_context for future messages
/privacy private    → Disable cross_context
/forget [topic]     → Delete memories about topic (TBD: how to implement)
/mymemories         → Show what agent knows (paginated)
```

### Opt-in Phrases (detected by agent)
```
"you can remember this"     → mark as cross_context
"this is private"           → mark as private
"forget that"               → soft delete last memory
"don't share this"          → mark as private
```

---

## LLM Integration Points

Lugares donde se necesita LLM (modelo TBD):

| Component | Purpose | Trigger |
|-----------|---------|---------|
| User summary generation | Consolidate user knowledge | Periodic / on-demand |
| Group topic detection | Understand what group is about | On join + periodic |
| Memory extraction | Convert messages to atomic memories | Each message |
| Privacy intent detection | Detect "this is private" phrases | Each message |
| Identity linking hints | Detect "my twitter is @x" | Each message |

---

## Migration Path

### Phase 1: Add fields to existing schema
- Add `group_id`, `context_type`, `privacy_scope` to memories
- Default: `group_id=null`, `context_type="dm"`, `privacy_scope="private"`
- No breaking changes

### Phase 2: Create user_profiles table
- Start collecting identifiers
- Link existing user_ids to profiles
- Generate initial summaries

### Phase 3: Create group_contexts table
- Register groups where agent participates
- Generate group summaries

### Phase 4: Implement privacy filters
- Update retrieval to respect privacy_scope
- Add user commands

---

## Open Questions

1. **¿Cuánto tiempo retener memorias de grupos?**
   - Forever? 90 days? Configurable per group?

2. **¿Qué pasa si usuario sale del grupo?**
   - ¿Mantener memorias? ¿Archivar? ¿Borrar?

3. **¿Cómo manejar grupos públicos vs privados?**
   - Telegram tiene grupos públicos (link público)
   - ¿Diferente privacy_scope default?

4. **¿Rate limiting de memorias por grupo?**
   - Grupo muy activo = muchas memorias
   - ¿Sampling? ¿Solo mensajes importantes?

5. **¿Cómo detectar "mensajes importantes" en grupo?**
   - ¿Menciones al agente?
   - ¿Mensajes con entidades relevantes?
   - ¿LLM classifica importancia? (modelo TBD)

---

## Next Steps

- [ ] Finalizar schema fields
- [ ] Implementar group_id y context_type en vector_store
- [ ] Crear tabla user_profiles
- [ ] Crear tabla group_contexts
- [ ] Implementar privacy filtering en retrieval
- [ ] Definir modelos LLM para cada task (TBD)
- [ ] User commands para privacy
