# Plan de Integracion: a0x-memory con services-backend

## 1. Analisis del Sistema Actual (Zep Cloud)

### 1.1 Estructura del zep-service

El `services-backend` utiliza Zep Cloud como sistema de memoria a traves de un paquete dedicado:

**Ubicacion:** `/home/oydual3/a0x/services-backend/packages/zep-service/`

**Componentes principales:**
- `ZepService` - Wrapper del cliente `@getzep/zep-cloud`
- `IntelligentSearchService` - Busqueda semantica avanzada

**Funcionalidades que provee:**
1. **User Management:** `addUser()`, `getUser()`, `updateUser()`, `deleteUser()`
2. **Thread Management:** `createThread()`, `getThread()`, `addMessages()`, `addMessageBatch()`
3. **Memory Operations:** `getMemory()`, `getUserContext()`
4. **Graph Operations:** `addToGraph()`, `searchUserGraph()`, `searchGroupGraph()`, `addFactTriple()`
5. **Episodes:** `getEpisodesByUserId()`, `getRecentEpisodes()`

### 1.2 Integracion con Diamond Orchestrator

El Diamond Orchestrator usa Zep en **4 facets clave:**

| Facet | Zone | Funcion | Archivo |
|-------|------|---------|---------|
| `EarlyMemoryFacet` | BOOTSTRAP | Carga historial inicial para routing | `bootstrap/early-memory.facet.ts` |
| `MemoryFacet` | ENRICH | Recupera memoria semantica | `enrich/memory.facet.ts` |
| `MemoryWriterFacet` | CLEANUP | Escribe conversacion a memoria | `cleanup/memory-writer.facet.ts` |
| `SpamDetectionFacet` | CLEANUP | Deteccion de spam con Zep | `cleanup/spam-detection.facet.ts` |

### 1.3 Processors de Memoria

**IntelligentShortTermMemoryProcessor** (`intelligent-short-term-memory.processor.ts`)
- Combina memoria reciente, semantica por query de usuario, semantica por respuesta del agente, y episodios importantes
- Usa `ZepQueryOptimizer` para queries de 400 chars max
- Devuelve: `history`, `semanticMemoryUser`, `semanticMemoryAgent`, `importantMemory`

**ZepMemoryWriterProcessor** (`zep-memory-writer.processor.ts`)
- Fire-and-forget en background
- Guarda user message + assistant response
- Crea user/thread si no existe (manejo de 404)
- Guarda action results al graph

---

## 2. Comparacion: Zep Cloud vs a0x-memory

| Caracteristica | Zep Cloud | a0x-memory |
|----------------|-----------|------------|
| **Arquitectura** | Servicio cloud managed | Self-hosted (Python/FastAPI) |
| **Almacenamiento** | Cloud propietario | LanceDB (local/VM) + Firestore |
| **Modelo de threads** | User -> Threads -> Messages | Agent -> Groups/DMs -> Messages |
| **Privacy model** | Por usuario | DM/Group con `is_shareable` flag |
| **Knowledge Graph** | Neo4j-based | LanceDB vector tables |
| **User Profiles** | Graph-based facts | LLM-generated profiles + evidence-based facts |
| **Group Support** | Limitado (no disponible en v3) | Nativo (group_memories, user_memories) |
| **Summarization** | Automatico (episodio) | Jerarquico (micro/chunk/block/era) |
| **Spam Detection** | Manual | Embedding similarity automatico |
| **Costo** | Por uso API | Self-hosted (infra) |

### 2.1 Ventajas de a0x-memory

1. **Privacy-first:** `is_shareable` flag para DM memories
2. **Group-native:** Soporte de primer nivel para grupos (Telegram, etc)
3. **Evidence-based facts:** Facts con confidence y evidence count
4. **Hierarchical summaries:** Micro -> Chunk -> Block -> Era
5. **Cross-context profiles:** User profiles agregados de DMs y grupos
6. **Adaptive thresholds:** Procesamiento dinamico basado en actividad
7. **Local embeddings:** `intfloat/multilingual-e5-small` (384 dims)

### 2.2 Endpoints de a0x-memory vs Zep

| Operacion | Zep | a0x-memory |
|-----------|-----|------------|
| Agregar mensaje pasivo | N/A | `POST /v1/memory/passive` |
| Agregar mensaje + contexto | `addMessages()` | `POST /v1/memory/active` |
| Obtener contexto | `getMemory()` + `getUserContext()` | `POST /v1/memory/context` |
| Buscar en graph | `searchUserGraph()` | (incluido en context) |
| User profile | `getUser()` | `GET /v1/profiles/user/{id}` |
| Group profile | N/A | `GET /v1/profiles/group/{id}` |
| User facts | `getUserContext()` | `GET /v1/facts/{user_id}` |
| Summaries | (automatico) | `GET /v1/dm/{id}/summaries`, `GET /v1/groups/{id}/summaries` |

---

## 3. Estructura Propuesta del Nuevo Servicio

### 3.1 Estructura del Paquete

```
services-backend/packages/a0x-memory-service/
├── src/
│   ├── index.ts                    # Exports principales
│   ├── types/
│   │   ├── index.ts               # Re-exports
│   │   ├── memory.types.ts        # Tipos de memoria
│   │   ├── profile.types.ts       # Tipos de perfil
│   │   └── platform-identity.ts   # Mapeo de plataformas
│   ├── services/
│   │   ├── a0x-memory.service.ts  # Servicio principal
│   │   ├── context.service.ts     # Retrieval de contexto
│   │   └── profile.service.ts     # Gestion de perfiles
│   └── utils/
│       ├── identity-mapper.ts     # Mapeo PlatformIdentity -> a0x-memory format
│       └── response-formatter.ts  # Formateo de respuestas
├── package.json
├── tsconfig.json
└── README.md
```

### 3.2 Interfaz Principal (A0xMemoryService)

```typescript
// services-backend/packages/a0x-memory-service/src/services/a0x-memory.service.ts

export interface A0xMemoryServiceConfig {
  baseUrl: string;  // http://136.118.160.81:8080
  timeout?: number;
}

export class A0xMemoryService {
  constructor(config: A0xMemoryServiceConfig);

  // ==================== MEMORY INGESTION ====================

  /**
   * Agregar mensaje pasivo (fire-and-forget)
   * El sistema decide cuando procesar batch
   */
  async addPassiveMemory(request: PassiveMemoryRequest): Promise<PassiveMemoryResponse>;

  /**
   * Agregar mensaje Y obtener contexto inmediatamente
   * Para cuando el agente necesita responder
   */
  async addActiveMemory(request: ActiveMemoryRequest): Promise<ActiveMemoryResponse>;

  // ==================== CONTEXT RETRIEVAL ====================

  /**
   * Obtener contexto para una query (RAG)
   * SIN agregar mensaje
   */
  async getContext(request: ContextRequest): Promise<ContextResponse>;

  /**
   * Obtener memoria formateada (Zep-compatible)
   */
  async getMemory(threadId: string, options?: MemoryOptions): Promise<A0xMemory>;

  // ==================== PROFILES ====================

  /**
   * Obtener perfil global del usuario
   */
  async getUserProfile(userId: string, agentId: string): Promise<UserProfile | null>;

  /**
   * Obtener perfil del grupo
   */
  async getGroupProfile(groupId: string, agentId: string): Promise<GroupProfile | null>;

  /**
   * Obtener perfil del usuario en un grupo especifico
   */
  async getUserInGroupProfile(userId: string, groupId: string, agentId: string): Promise<UserInGroupProfile | null>;

  // ==================== FACTS ====================

  /**
   * Obtener facts verificados del usuario
   */
  async getUserFacts(userId: string, agentId: string): Promise<UserFacts>;

  // ==================== SUMMARIES ====================

  /**
   * Obtener summaries de DM
   */
  async getDMSummaries(userId: string, agentId: string): Promise<SummaryHierarchy>;

  /**
   * Obtener summaries de grupo
   */
  async getGroupSummaries(groupId: string, agentId: string): Promise<SummaryHierarchy>;

  // ==================== ADMIN ====================

  /**
   * Estadisticas del agente
   */
  async getStats(agentId: string): Promise<AgentStats>;

  /**
   * Reset completo (solo para testing)
   */
  async reset(agentId: string, confirm: boolean): Promise<void>;
}
```

### 3.3 Tipos Principales

```typescript
// services-backend/packages/a0x-memory-service/src/types/memory.types.ts

export interface PlatformIdentityMapping {
  platform: 'telegram' | 'xmtp' | 'farcaster' | 'twitter' | 'direct';
  telegramId?: number;
  walletAddress?: string;
  fid?: string;
  twitterId?: string;
  clientId?: string;
  username?: string;
  chatId?: string;  // Negativo = grupo, null/ausente = DM
}

export interface PassiveMemoryRequest {
  agentId: string;
  message: string;
  platformIdentity: PlatformIdentityMapping;
  speaker: string;
}

export interface PassiveMemoryResponse {
  success: boolean;
  isGroup: boolean;
  groupId: string | null;
  userId: string;
  processingScheduled: boolean;
  memoriesCreated: number;
  isSpam: boolean;
  isBlocked: boolean;
  spamScore: number;
}

export interface ContextRequest {
  agentId: string;
  query: string;
  platformIdentity: PlatformIdentityMapping;
  involvedUsers?: string[];
  includeRecent?: boolean;
  recentLimit?: number;
  memoryLimit?: number;
}

export interface ContextResponse {
  success: boolean;
  recentMessages: RecentMessage[];
  relevantMemories: RelevantMemory[];
  userProfile: UserProfile | null;
  groupProfile: GroupProfile | null;
  formattedContext: string;  // Listo para inyectar en prompt
}

export interface A0xMemory {
  messages: A0xMessage[];
  summary?: string;
  context?: string;
  facts?: string[];
  userProfile?: UserProfile;
  groupProfile?: GroupProfile;
}
```

---

## 4. Pasos de Implementacion

### Fase 1: Crear el paquete base (2-3 dias)

1. **Crear estructura del paquete**
   ```bash
   mkdir -p services-backend/packages/a0x-memory-service/src/{types,services,utils}
   ```

2. **Implementar tipos base**
   - Definir todas las interfaces y tipos
   - Mapeo entre `PlatformIdentity` (services-backend) y `platform_identity` (a0x-memory)

3. **Implementar `A0xMemoryService`**
   - HTTP client con axios
   - Manejo de errores
   - Timeout y retry logic

4. **Tests unitarios**
   - Mock del HTTP client
   - Tests de mapeo de identidad

### Fase 2: Crear procesadores (2 dias)

1. **A0xShortTermMemoryProcessor**
   - Equivalente a `IntelligentShortTermMemoryProcessor`
   - Usar `POST /v1/memory/context`
   - Devolver `history`, `semanticMemory`, `userProfile`, `groupProfile`

2. **A0xMemoryWriterProcessor**
   - Equivalente a `ZepMemoryWriterProcessor`
   - Usar `POST /v1/memory/passive`
   - Fire-and-forget en background

### Fase 3: Crear facets (1-2 dias)

1. **A0xMemoryFacet** (ENRICH zone)
   - Wrapper del processor
   - order = 231 (despues de MemoryFacet de Zep)

2. **A0xMemoryWriterFacet** (CLEANUP zone)
   - Wrapper del processor
   - order = 421 (despues de MemoryWriterFacet de Zep)

3. **A0xEarlyMemoryFacet** (BOOTSTRAP zone)
   - Para cargar historial inicial
   - order = 11 (despues de EarlyMemoryFacet de Zep)

### Fase 4: Integracion con Diamond Orchestrator (1-2 dias)

1. **Modificar `DiamondOrchestratorServices`**
   ```typescript
   export interface DiamondOrchestratorServices {
     // ... existing
     a0xMemoryService?: A0xMemoryService;  // NUEVO
   }
   ```

2. **Modificar `registerDefaultFacets()`**
   - Agregar condicion para registrar facets de a0x-memory
   - Usar feature flag o config para elegir Zep vs a0x-memory

3. **Config en agent-execution.ts**
   ```typescript
   const memoryProvider = process.env.MEMORY_PROVIDER || 'zep'; // 'zep' | 'a0x-memory'
   ```

### Fase 5: Testing e Integracion (2-3 dias)

1. **Tests de integracion**
   - Test completo del pipeline con a0x-memory
   - Comparacion de resultados vs Zep

2. **Testing en ambiente de desarrollo**
   - Deploy del a0x-memory-service en dev
   - Pruebas con jessexbt

3. **Documentacion**
   - README del nuevo paquete
   - Guia de migracion

---

## 5. Integracion con Diamond Orchestrator

### 5.1 Modificaciones al Orchestrator

```typescript
// services-backend/packages/agent-execution/src/diamond/diamond.orchestrator.ts

export interface DiamondOrchestratorServices {
  // ... existing services ...
  zepService?: ZepService;
  a0xMemoryService?: A0xMemoryService;  // NUEVO
}

private registerDefaultFacets(): void {
  // ...

  // BOOTSTRAP zone
  // Elegir entre Zep o a0x-memory para early memory
  const memoryProvider = process.env.MEMORY_PROVIDER || 'zep';

  if (memoryProvider === 'a0x-memory' && this.services.a0xMemoryService) {
    this.registerFacet(new A0xEarlyMemoryFacet(this.services.a0xMemoryService));
  } else if (this.services.zepService) {
    this.registerFacet(new EarlyMemoryFacet(this.services.zepService));
  }

  // ENRICH zone
  if (memoryProvider === 'a0x-memory' && this.services.a0xMemoryService) {
    this.registerFacet(new A0xMemoryFacet(this.services.a0xMemoryService));
  } else if (this.services.zepService) {
    this.registerFacet(new MemoryFacet(this.services.zepService));
  }

  // CLEANUP zone
  if (memoryProvider === 'a0x-memory' && this.services.a0xMemoryService) {
    this.registerFacet(new A0xMemoryWriterFacet(this.services.a0xMemoryService));
  } else if (this.services.zepService) {
    this.registerFacet(new MemoryWriterFacet(this.services.zepService));
  }

  // ...
}
```

### 5.2 Flujo con a0x-memory

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BOOTSTRAP                                                               │
│                                                                          │
│  A0xEarlyMemoryFacet:                                                   │
│  - POST /v1/memory/context con query="*" y limit=5                      │
│  - Obtiene ultimos mensajes para routing decisions                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  ENRICH (parallel)                                                       │
│                                                                          │
│  A0xMemoryFacet:                                                        │
│  - POST /v1/memory/context con query=userMessage                        │
│  - Obtiene:                                                             │
│    * recentMessages -> context.history                                  │
│    * relevantMemories -> context.semanticMemory                         │
│    * userProfile -> context.userProfile                                 │
│    * groupProfile -> context.groupProfile                               │
│    * formattedContext -> context.memoryContext                          │
│                                                                          │
│  [En paralelo con KnowledgeFacet, PersonalityFacet, etc.]               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CLEANUP (fire-and-forget)                                               │
│                                                                          │
│  A0xMemoryWriterFacet:                                                  │
│  - POST /v1/memory/passive con:                                         │
│    * message = userMessage                                              │
│    * platformIdentity = mapeo desde context.userIdentity                │
│    * speaker = username                                                 │
│  - Sistema procesa async cuando hay suficientes mensajes                │
│  - Genera memories, profiles, facts, summaries automaticamente          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Mapeo de Identidades

```typescript
// services-backend/packages/a0x-memory-service/src/utils/identity-mapper.ts

export function mapPlatformIdentityToA0x(
  platformIdentity: PlatformIdentity,
  agentId: string
): PlatformIdentityMapping {

  switch (platformIdentity.platform) {
    case Platform.TELEGRAM:
      const telegram = platformIdentity as TelegramIdentity;
      return {
        platform: 'telegram',
        telegramId: parseInt(telegram.telegramId),
        username: telegram.username,
        chatId: telegram.chatId ? String(telegram.chatId) : undefined,
        // chatId negativo = grupo, null = DM
      };

    case Platform.XMTP:
      const xmtp = platformIdentity as XMTPIdentity;
      return {
        platform: 'xmtp',
        walletAddress: xmtp.walletAddress,
        username: xmtp.ensName || xmtp.basename,
        // XMTP DMs don't have chatId (always DM)
      };

    case Platform.FARCASTER:
      const farcaster = platformIdentity as FarcasterIdentity;
      return {
        platform: 'farcaster',
        fid: farcaster.fid,
        username: farcaster.username,
        // Farcaster doesn't have groups yet
      };

    case Platform.TWITTER:
      const twitter = platformIdentity as TwitterIdentity;
      return {
        platform: 'twitter',
        twitterId: twitter.twitterId,
        username: twitter.username,
        // Twitter is always public (group-like behavior)
      };

    case Platform.DIRECT:
      const direct = platformIdentity as DirectIdentity;
      return {
        platform: 'direct',
        clientId: direct.clientId,
        username: direct.sessionId,
        // Direct API is always DM
      };

    default:
      throw new Error(`Unknown platform: ${(platformIdentity as any).platform}`);
  }
}
```

---

## 6. Plan de Testing

### 6.1 Tests Unitarios

```typescript
// packages/a0x-memory-service/src/__tests__/a0x-memory.service.test.ts

describe('A0xMemoryService', () => {
  describe('addPassiveMemory', () => {
    it('should handle Telegram DM', async () => { /* ... */ });
    it('should handle Telegram group', async () => { /* ... */ });
    it('should handle spam detection', async () => { /* ... */ });
  });

  describe('getContext', () => {
    it('should return formatted context for DM', async () => { /* ... */ });
    it('should include group profile in group context', async () => { /* ... */ });
    it('should handle missing user gracefully', async () => { /* ... */ });
  });

  describe('identity mapping', () => {
    it('should map Telegram identity correctly', async () => { /* ... */ });
    it('should map XMTP identity correctly', async () => { /* ... */ });
  });
});
```

### 6.2 Tests de Integracion

```typescript
// packages/agent-execution/src/__tests__/a0x-memory-integration.test.ts

describe('A0x Memory Integration', () => {
  describe('with Diamond Orchestrator', () => {
    it('should fetch memory in ENRICH zone', async () => { /* ... */ });
    it('should write memory in CLEANUP zone', async () => { /* ... */ });
    it('should handle group context', async () => { /* ... */ });
  });

  describe('comparison with Zep', () => {
    it('should produce similar context format', async () => { /* ... */ });
    it('should handle same user across providers', async () => { /* ... */ });
  });
});
```

### 6.3 Test Manual con jessexbt

1. **Preparar ambiente de testing**
   ```bash
   export MEMORY_PROVIDER=a0x-memory
   export A0X_MEMORY_URL=http://136.118.160.81:8080
   ```

2. **Crear test conversation**
   - Enviar mensajes en Telegram
   - Verificar que se guarden en a0x-memory
   - Verificar que el contexto se recupere correctamente

3. **Verificar integracion**
   - Comparar respuestas con Zep
   - Verificar profiles y facts

---

## 7. Consideraciones de Migracion

### 7.1 Feature Flag Approach

La integracion se puede hacer gradualmente usando feature flags:

```typescript
// Config environment variables
MEMORY_PROVIDER=zep           # Default: usa Zep
MEMORY_PROVIDER=a0x-memory    # Usa a0x-memory
MEMORY_PROVIDER=hybrid        # Usa ambos (escribe a ambos, lee de a0x-memory)
```

### 7.2 Migracion de Datos

Si se quiere migrar memoria existente de Zep a a0x-memory:

1. **Export de Zep**
   - Usar `getMemory()` y `getEpisodesByUserId()` para cada usuario
   - Export a formato JSON

2. **Import a a0x-memory**
   - Usar `POST /v1/memory/passive` para cada mensaje
   - Dejar que el sistema procese y genere memories/profiles

### 7.3 Rollback Plan

Si hay problemas:
1. Cambiar `MEMORY_PROVIDER=zep`
2. Reiniciar servicios
3. El sistema vuelve a usar Zep inmediatamente

---

## 8. Timeline Estimado

| Fase | Duracion | Dependencias |
|------|----------|--------------|
| Fase 1: Paquete base | 2-3 dias | - |
| Fase 2: Procesadores | 2 dias | Fase 1 |
| Fase 3: Facets | 1-2 dias | Fase 2 |
| Fase 4: Orchestrator | 1-2 dias | Fase 3 |
| Fase 5: Testing | 2-3 dias | Fase 4 |
| **Total** | **8-12 dias** | - |

---

## 9. Archivos Criticos para Implementacion

Los archivos mas criticos para implementar este plan:

- `/home/oydual3/a0x/services-backend/packages/zep-service/src/services/zep.service.ts` - Patron a seguir para el nuevo servicio
- `/home/oydual3/a0x/services-backend/packages/agent-execution/src/processors/intelligent-short-term-memory.processor.ts` - Logica de memoria a replicar
- `/home/oydual3/a0x/services-backend/packages/agent-execution/src/diamond/diamond.orchestrator.ts` - Donde integrar los nuevos facets
- `/home/oydual3/a0x/services-backend/packages/agent-execution/src/diamond/facets/enrich/memory.facet.ts` - Patron de facet a seguir
- `/home/oydual3/a0x/a0x-memory/docs/API_REFERENCE.md` - Documentacion de endpoints del sistema de memoria
