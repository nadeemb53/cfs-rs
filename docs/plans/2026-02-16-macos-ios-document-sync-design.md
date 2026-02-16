# CP Document Intelligence Sync - Design

**Date:** 2026-02-16
**Status:** Draft
**Target:** Mac-iOS document sync with Q&A

---

## 1. Overview

Build a document intelligence sync system where:
- **Mac**: Users explicitly select documents → indexed → synced to cloud relay
- **iOS**: Users pull documents → query locally → get answers with citations

**Key Constraints:**
- Relay server hosted by us
- Device pairing via short code (Ed25519 key exchange)
- Inline citations + document list for verification

---

## 2. Architecture

### 2.1 System Components

```
┌─────────────────┐      ┌─────────────┐      ┌─────────────────┐
│   macOS App     │ ──── │  Relay      │ ──── │   iOS App       │
│                 │      │  Server     │      │                 │
│ • File picker   │      │             │      │ • Query UI      │
│ • Indexer       │      │ /pair       │      │ • Citations     │
│ • Sync client   │      │ /push       │      │ • Local index   │
│ • Local index   │      │ /pull       │      │ • Context asm   │
└─────────────────┘      └─────────────┘      └─────────────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                          ┌──────┴──────┐
                          │  cp-core    │
                          │  cp-graph   │
                          │  cp-query   │
                          │  cp-sync    │
                          │  cp-embed   │
                          └─────────────┘
```

### 2.2 Data Flow

**Mac (Index & Push):**
1. User selects files via NSOpenPanel
2. Parse documents (PDF, Markdown, text) using `cp-parser`
3. Generate chunks with byte offsets
4. Compute content-based IDs (BLAKE3-16)
5. Generate embeddings using `cp-embeddings` (fast-path)
6. Store in local SQLite + HNSW (`cp-graph`)
7. Compute cognitive diff vs last sync
8. Encrypt diff (XChaCha20-Poly1305)
9. Sign with Ed25519
10. Push to relay server

**iOS (Pull & Query):**
1. Connect to relay, authenticate with device keys
2. Pull all diffs since last known sequence
3. Decrypt and verify signatures
4. Apply to local state (CRDT merge)
5. Rebuild local index from canonical state
6. User types query
7. Generate query embedding
8. Hybrid search (semantic + FTS5)
9. RRF fusion with integer scoring
10. Assemble context with citations
11. Display answer + sources

---

## 3. Relay Server

### 3.1 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/pair/init` | POST | Initiate pairing (device publishes public key) |
| `/pair/confirm` | POST | Confirm pairing (device B receives A's key) |
| `/pair/status` | GET | Check pairing status |
| `/push` | POST | Upload encrypted diff (requires auth) |
| `/pull` | GET | Pull diffs since sequence (requires auth) |
| `/acknowledge` | DELETE | Acknowledge receipt |
| `/roots` | GET | List state roots |

### 3.2 Device Pairing Protocol

```
Device A (Mac)                          Device B (iOS)
    │                                        │
    │──── POST /pair/init {pubkey, name} ────│
    │                                        │
    │                                        │ (display pairing code)
    │                                        │
    │──── POST /pair/confirm {pubkey} ──────►│
    │         (or code verification)         │
    │                                        │
    │<─── OK ────────────────────────────────│
    │                                        │
    │    Both devices now have:              │
    │    - Their own keypair                 │
    │    - Peer's public key                 │
    │    - Shared secret (via X25519)        │
```

### 3.3 Authentication

- Each device has Ed25519 keypair
- Device public key registered with relay
- Requests signed with device's private key
- Header: `X-Device-ID`: hex(device_id)
- Header: `X-Device-Signature`: hex(signature(request_body))

### 3.4 Storage Schema

```sql
-- Devices table
CREATE TABLE devices (
    device_id TEXT PRIMARY KEY,
    public_key TEXT NOT NULL,
    name TEXT,
    created_at INTEGER NOT NULL,
    last_seen INTEGER NOT NULL
);

-- Pairings (which devices can sync with each other)
CREATE TABLE pairings (
    device_a TEXT NOT NULL,
    device_b TEXT NOT NULL,
    shared_secret TEXT,  -- encrypted X25519 shared secret
    paired_at INTEGER NOT NULL,
    PRIMARY KEY (device_a, device_b)
);

-- Encrypted diffs (per CP-013)
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

-- State roots for sync verification
CREATE TABLE state_roots (
    device_id TEXT NOT NULL,
    root_hash TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    PRIMARY KEY (device_id, sequence)
);
```

---

## 4. macOS Desktop App

### 4.1 UI Components

**Main Window (NSWindowController)**
- Toolbar: Sync status, Add Documents, Settings
- Sidebar: Document list (grouped by folder)
- Content: Selected document preview
- Bottom bar: Last sync time, document count

**Document Picker**
- NSOpenPanel with multi-select
- Filter: PDF, Markdown, Text
- Option to include subfolders

**Settings Sheet**
- Relay server URL
- Device name
- Pairing status
- Sync interval

### 4.2 Document Processing Pipeline

```
File → Parser → Chunker → Embedder → Storage
           │          │          │
           │          │          └─ cp-embeddings (fast-path)
           │          └──────────── cp-graph (SQLite + HNSW)
           └──────────────────────── cp-parser
```

**Processing Steps:**
1. Read file bytes
2. Detect MIME type (extension-based for now)
3. Parse content:
   - PDF: Extract text using pdf-extract
   - Markdown: Parse to plain text
   - Text: Use as-is
4. Chunk with 512-byte target, 64-byte overlap
5. Canonicalize text (NFC, LF, trim)
6. Generate chunk IDs: BLAKE3(doc_id || sequence)[0..16]
7. Generate embeddings: cp-embeddings (384-dim, i16)
8. Store in cp-graph

### 4.3 Sync Flow

```
Local State ──diff──► Encrypt ──sign──► Push to Relay
                           │
                    XChaCha20-Poly1305
                    Ed25519 signature
```

**Sync Trigger:**
- Manual: User clicks "Sync"
- Auto: Every 5 minutes (configurable)

### 4.4 Key Modules

**DocumentIndexer**
- `add_documents(paths: Vec<PathBuf>) -> Result<()>`
- `remove_document(doc_id: DocId) -> Result<()>`
- `get_document_count() -> usize`

**SyncManager**
- `sync() -> Result<SyncResult>`
- `push_diff(diff: SignedDiff) -> Result<()>`
- `get_last_sync_time() -> Option<HlcTimestamp>`

**RelayConnection**
- `pair_device(server_url: &str) -> Result<PairingCode>`
- `confirm_pairing(code: &str) -> Result<()>`
- `is_paired() -> bool`

---

## 5. iOS Mobile App

### 5.1 UI Components

**Main View (SwiftUI)**
- Navigation: Document list → Query view
- Query view: Text input + answer display
- Sources: Expandable citation list

**Query Interface**
```
┌─────────────────────────────────────┐
│  Ask about your documents           │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │ What is the project timeline?│   │
│  └─────────────────────────────┘   │
│                                     │
│  [Ask]                             │
├─────────────────────────────────────┤
│  Answer with citations...           │
│                                     │
│  According to the project plan...    │
│                                     │
│  Sources:                          │
│  ┌─────────────────────────────┐   │
│  │ 📄 project-plan.pdf (92%)  │   │
│  │ 📄 roadmap.md (87%)        │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 5.2 Pull & Merge

**Pull Flow:**
1. Connect to relay, authenticate
2. Request diffs since last_sequence
3. Decrypt each diff with shared secret
4. Verify sender signature
5. Apply cognitive diff to local state
6. Rebuild index from canonical state
7. Update last_sequence

**Conflict Resolution:**
- Per CP-013: LWW with HLC timestamps
- Each device has unique node_id
- Merge: keep highest HLC

### 5.3 Query Pipeline

```
Query → Embed → Hybrid Search → RRF → Context → Answer
         │           │              │       │
         │           │              │    cp-context
         │           │              │
         │           └──────────────┘
         │                   │
         │            cp-query
         │            (RRF fusion)
         │
         └───────────── cp-embeddings
```

**Citation Extraction:**
1. Get top-K chunks (K=10)
2. For each chunk, extract n-gram overlap with answer
3. Mark overlapping spans in source
4. Display with highlighting

### 5.4 Key Modules

**SyncEngine**
- `pull() -> Result<Vec<SignedDiff>>`
- `merge_diffs(diffs: Vec<SignedDiff>) -> Result<()>`
- `rebuild_index() -> Result<()>`

**QueryService**
- `query(question: &str) -> Result<Answer>`
- `get_sources(doc_ids: Vec<DocId>) -> Vec<Source>`

**ContextAssembler**
- `assemble(query: &str, chunks: Vec<Chunk>, budget: usize) -> Context`
- Per CP-020: Token budgeting, citation preservation

---

## 6. Security

### 6.1 Encryption

- **Transit**: HTTPS/TLS to relay
- **Storage**: XChaCha20-Poly1305 encrypted diffs
- **Keys**:
  - Device Ed25519 keypair (identity)
  - X25519 keypair (for shared secret derivation)
  - Shared secret derived via X25519 ECDH

### 6.2 Authentication

- Device public key registered with relay
- Requests signed with Ed25519 private key
- Signature in header: `X-Device-Signature`

### 6.3 Verification

- Each diff signed by sender
- Recipient verifies before applying
- State root provides Merkle commitment
- cp-canonical-embeddings for deterministic verification

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Document parsing (PDF, Markdown, text)
- Chunking (boundaries, overlap)
- Embedding generation (determinism)
- Sync merge (CRDT properties)
- Query retrieval (RRF scoring)

### 7.2 Integration Tests

- Mac → Relay → iOS sync flow
- Conflict resolution
- Citation extraction accuracy

### 7.3 Determinism Tests

- Embedding generation across runs
- RRF scoring consistency
- Merkle root computation

---

## 8. Implementation Priority

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | Relay pairing | Device registration, pairing protocol |
| 2 | Mac document picker | File selection, parsing, indexing |
| 3 | Mac sync client | Push to relay |
| 4 | iOS sync engine | Pull from relay, merge |
| 5 | iOS query UI | Search, citations |
| 6 | E2E verification | Full flow test |

---

## 9. Open Questions

1. **Query model**: Local LLM on iOS (llama.cpp) or API?
2. **Document updates**: How to detect changed files on Mac?
3. **Deletion sync**: How to sync document deletions?
4. **Relay scaling**: Single server or distributed?

---

## 10. References

- CP-001: Data Model & Identity
- CP-002: Storage Engine
- CP-003: Determinism
- CP-010: Embedding Protocol
- CP-011: Indexing Protocol
- CP-012: Retrieval Protocol
- CP-013: Sync Protocol
- CP-020: Intelligence & Context
