# Mac-iOS Document Intelligence Sync Design

**Date:** 2026-02-16
**Status:** Draft
**Target:** Production deployment

## Overview

This document describes the architecture for a document intelligence sync system where users can select documents on macOS, have them indexed and synced to a cloud relay, and query them on iOS with verifiable citations.

## Requirements Summary

| Requirement | Implementation |
|-------------|----------------|
| Document selection on Mac | Explicit file/folder picker |
| Document sync | Cloud relay server (we host) |
| Authentication | Device pairing code (Ed25519 key exchange) |
| Query on iOS | Hybrid search (semantic + lexical) |
| Verification | Inline citations + document list with scores |

## Architecture

### System Components

```
┌─────────────────┐      ┌─────────────┐      ┌─────────────────┐
│   macOS App     │      │  Relay      │      │   iOS App       │
│                 │      │  Server     │      │                 │
│ • Document      │ ──── │             │ ──── │ • Query UI      │
│   picker        │      │ • Device    │      │ • Citations     │
│ • Indexer       │      │   pairing   │      │ • Local index   │
│ • Sync client   │      │ • Diff      │      │ • Sync client   │
│                 │      │   storage   │      │                 │
└─────────────────┘      └─────────────┘      └─────────────────┘
```

### Data Flow

**Mac → iOS Sync:**
1. User selects documents via file picker
2. Documents parsed and chunked (`cp-parser`)
3. Chunks hashed to create stable IDs (`cp-core`)
4. Embeddings generated (`cp-embeddings` for fast-path, `cp-canonical-embeddings` for verification)
5. Cognitive diff computed (`cp-sync`)
6. Diff encrypted with recipient's public key
7. Uploaded to relay server
8. iOS device pulls and applies diff

**Query on iOS:**
1. User enters query
2. Query embedded using local model
3. Hybrid search: HNSW (semantic) + FTS5 (lexical)
4. Results fused using integer RRF
5. Context assembler builds context with token budget
6. LLM generates answer with citations
7. Citations extracted via n-gram overlap

## Component Specifications

### 1. Relay Server (cp-relay-server)

**Current State:** Basic push/pull implemented, needs device pairing

**Required Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/pair/init` | POST | Initiate pairing (sender device) |
| `/pair/respond` | POST | Respond to pairing (receiver device) |
| `/pair/confirm` | POST | Confirm pairing complete |
| `/push` | POST | Upload encrypted diff |
| `/pull` | GET | Pull diffs since sequence |
| `/ack | Acknowledge` | DELETEnowledge receipt |
| `/devices` | GET | List paired devices |

**Device Pairing Protocol:**

```
Device A (Mac)                           Device B (iOS)
    |                                          |
    |---- POST /pair/init (A's pubkey) ------>|
    |<--- 200 OK (relay assigns pairing_id) ---|
    |                                          |
    |         [User copies code from Mac]      |
    |                                          |
    |---- POST /pair/respond (pairing_id, ----|
    |      B's pubkey)                        |
    |<--- 200 OK (confirms receipt) ----------|
    |                                          |
    |         [User confirms on iOS]           |
    |                                          |
    |---- POST /pair/confirm (pairing_id) --->|
    |<--- 200 OK (both devices paired) --------|
```

**Storage Schema:**

```sql
-- Devices table
CREATE TABLE devices (
    device_id TEXT PRIMARY KEY,
    public_key TEXT NOT NULL,
    display_name TEXT,
    created_at INTEGER NOT NULL,
    last_seen INTEGER
);

-- Pairings (device-to-device trust)
CREATE TABLE pairings (
    id INTEGER PRIMARY KEY,
    device_a TEXT NOT NULL,
    device_b TEXT NOT NULL,
    shared_secret_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (device_a) REFERENCES devices(device_id),
    FOREIGN KEY (device_b) REFERENCES devices(device_id)
);

-- Encrypted diffs (existing)
CREATE TABLE diffs (
    sender_device_id TEXT NOT NULL,
    target_device_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    ciphertext TEXT NOT NULL,
    nonce TEXT NOT NULL,
    signature TEXT NOT NULL,
    sender_public_key TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sender_device_id, target_device_id, sequence)
);
```

**Authentication:**
- Bearer token per device (issued during pairing)
- Token = HMAC-SHA256(pairing_secret, device_id || timestamp)
- Short pairing code: 6 alphanumeric characters (displayed on Mac, entered on iOS)

### 2. macOS Desktop App (cp-desktop)

**UI Components:**

1. **Document Picker View**
   - System file/folder picker (NSOpenPanel)
   - Display selected paths with icons
   - Remove/re-add documents
   - Save selection to config (JSON)

2. **Sync Status View**
   - Last sync timestamp
   - Sync progress (documents indexed, uploaded)
   - List of paired devices
   - Manual sync trigger button

3. **Settings View**
   - Relay server URL
   - Device display name
   - Pairing code display
   - Index location

**Document Selection Storage:**

```json
// ~/.config/cp/documents.json
{
  "version": 1,
  "relay_url": "https://relay.example.com",
  "device_id": "abc123...",
  "selected_paths": [
    "/Users/nadeem/Documents/project-alpha",
    "/Users/nadeem/Downloads/paper.pdf"
  ],
  "exclude_patterns": ["*.tmp", ".git/*"],
  "last_full_sync": 1708000000
}
```

**Indexing Pipeline:**

```
File → Read → Parser (PDF/MD/TXT) → Chunks → Hash IDs → Embed → Store
```

1. Read file from selected path
2. Parse using `cp-parser` (supports PDF, Markdown, plain text)
3. Chunk with byte offsets (per CP-011)
4. Compute stable IDs using BLAKE3-16 of content
5. Generate embeddings using `cp-embeddings` (fast-path)
6. Store in local SQLite + HNSW (using `cp-graph`)
7. Compute state root (Merkle commitment)
8. Create cognitive diff
9. Encrypt with paired device's public key
10. Push to relay

### 3. iOS Mobile App (cp-mobile)

**UI Components:**

1. **Onboarding View**
   - Welcome screen
   - "Add Device" button → show pairing code input
   - Connect to relay prompt

2. **Document List View**
   - List of synced documents
   - Last updated timestamp
   - Pull-to-refresh to sync

3. **Query View**
   - Search bar
   - Results with inline citations
   - Source document list with scores

4. **Settings View**
   - Relay server URL
   - Device name
   - Pairing display

**Local Storage:**

- SQLite database for documents/chunks (via `cp-graph`)
- HNSW vector index for embeddings
- State root stored for sync verification
- Query cache with invalidation

**Query Response Format:**

```json
{
  "answer": "The main benefit is improved performance...",
  "citations": [
    {
      "chunk_id": "abc123",
      "document": "paper.pdf",
      "passage": "The main benefit is improved performance...",
      "score": 0.85
    },
    {
      "chunk_id": "def456",
      "document": "notes.md",
      "passage": "Performance improvements include...",
      "score": 0.72
    }
  ],
  "sources": [
    {"path": "paper.pdf", "score": 0.85},
    {"path": "notes.md", "score": 0.72}
  ]
}
```

### 4. Context Assembler (cp-query)

The context assembler (already implemented in `cp-query`) handles:

- Token budget management (configurable, default 4096)
- Deterministic context hash for verification
- Hallucination detection via phrase checking
- Citation extraction using n-gram overlap

**Usage in iOS App:**

```rust
let context = ContextAssembler::new(4096)
    .with_documents(&chunks)
    .with_query(&query)
    .assemble()?;

let answer = llm.generate(&context, &query);
let citations = CitationExtractor::extract(&answer, &chunks);
```

## Security Considerations

1. **Encryption:** XChaCha20-Poly1305 per CP-013
2. **Signing:** Ed25519 for device authentication and diff signing
3. **Key Storage:** Keychain (iOS), Keychain (macOS)
4. **Pairing Code:** 6-char alphanumeric, expires after 5 minutes
5. **Relay Trust:** Device public keys pinned after pairing

## Testing Strategy

1. **Unit Tests:** All CP crates (existing, 250+ tests)
2. **Integration Tests:**
   - Device pairing flow
   - Document sync Mac → iOS
   - Query with citations
3. **Determinism Tests:**
   - Embedding quantization across platforms
   - RRF ranking consistency
   - Merkle root computation

## Open Questions

1. **LLM Inference:** Run locally on device or use external API?
   - Recommendation: Start with external API (OpenAI/Anthropic), add local later

2. **File Types:** Start with PDF, Markdown, plain text. Add more later.

3. **Chunk Size:** Default 512 bytes, configurable?

4. **Sync Frequency:** On-change (FSEvents) or periodic?
   - Recommendation: Both - FSEvents for Mac, periodic pull for iOS

## Implementation Priority

1. **Phase 1:** Relay server device pairing + basic push/pull
2. **Phase 2:** macOS app document selection + indexing + sync
3. **Phase 3:** iOS app document pull + local index
4. **Phase 4:** Query UI with citations
5. **Phase 5:** Full verification (canonical embeddings)

## Related Documents

- CP-001: Data Model & Identity
- CP-010: Embedding Protocol
- CP-011: Indexing Protocol
- CP-012: Retrieval Protocol
- CP-013: Sync Protocol
- CP-020: Intelligence Protocol
