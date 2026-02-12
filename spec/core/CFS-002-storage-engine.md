# CFS-002: Storage Engine

> **Spec Version**: 1.0.0
> **Author**: Nadeem Bhati
> **Category**: Core
> **Requires**: CFS-001

## Synopsis

This specification defines the hybrid storage architecture used by CFS, combining SQLite for structured data persistence with HNSW (Hierarchical Navigable Small World) graphs for vector similarity search.

## Motivation

CFS requires a storage engine that provides:

1. **Durability**: Data persists across restarts
2. **ACID Transactions**: Atomic updates for consistency
3. **Efficient Vector Search**: Sub-linear similarity queries
4. **Full-Text Search**: Keyword-based retrieval
5. **Portability**: Works on desktop and mobile platforms

The hybrid SQLite + HNSW architecture achieves all these goals.

## Technical Specification

### 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GraphStore                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │       SQLite DB         │   │      HNSW Index         │  │
│  ├─────────────────────────┤   ├─────────────────────────┤  │
│  │ - documents             │   │ - Disk-based (Mmap)     │  │
│  │ - chunks                │   │ - Vector similarity     │  │
│  │ - embeddings            │   │ - Persisted independently│ │
│  │ - edges                 │   │ - Validated via Merkle  │  │
│  │ - state_roots           │   │                         │  │
│  │ - fts_chunks (FTS5)     │   │                         │  │
│  └─────────────────────────┘   └─────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. SQLite Schema

#### Documents Table

```sql
CREATE TABLE documents (
    id              BLOB PRIMARY KEY,     -- 16 bytes (BLAKE3-16)
    path            TEXT NOT NULL,
    content_hash    BLOB NOT NULL,        -- BLAKE3 hash (32 bytes)
    hierarchical_hash BLOB NOT NULL,      -- Merkle root of chunks
    mtime           INTEGER NOT NULL,     -- Unix timestamp (ms)
    size_bytes      INTEGER NOT NULL,
    mime_type       TEXT NOT NULL,
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000),
    updated_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);

CREATE INDEX idx_documents_path ON documents(path);
CREATE INDEX idx_documents_content_hash ON documents(content_hash);
```

#### Chunks Table

```sql
CREATE TABLE chunks (
    id              BLOB PRIMARY KEY,     -- 16 bytes (BLAKE3-16)
    document_id     BLOB NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text            TEXT NOT NULL,
    text_hash       BLOB NOT NULL,        -- BLAKE3 hash (32 bytes)
    byte_offset     INTEGER NOT NULL,
    byte_length     INTEGER NOT NULL,
    sequence        INTEGER NOT NULL,
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000),

    UNIQUE(document_id, sequence)
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_text_hash ON chunks(text_hash);
```

#### Embeddings Table

```sql
CREATE TABLE embeddings (
    id              BLOB PRIMARY KEY,     -- 16 bytes (BLAKE3-16)
    chunk_id        BLOB NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    vector          BLOB NOT NULL,        -- i16 array, serialized
    model_hash      BLOB NOT NULL,        -- BLAKE3 hash (32 bytes)
    l2_norm         REAL NOT NULL,        -- Precomputed for cosine similarity
    dimension       INTEGER NOT NULL,
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);

CREATE INDEX idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX idx_embeddings_model_hash ON embeddings(model_hash);
```

#### Edges Table

```sql
CREATE TABLE edges (
    source_id       BLOB NOT NULL,        -- 16 bytes (BLAKE3-16)
    target_id       BLOB NOT NULL,        -- 16 bytes (BLAKE3-16)
    kind            TEXT NOT NULL,        -- "DocToChunk", "ChunkToEmbedding"
    weight          REAL,                 -- Optional relationship strength
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000),

    PRIMARY KEY (source_id, target_id, kind)
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
```

#### State Roots Table

```sql
CREATE TABLE state_roots (
    sequence        INTEGER PRIMARY KEY,  -- Monotonic sequence number
    hash            BLOB NOT NULL UNIQUE, -- BLAKE3 Merkle root
    parent_hash     BLOB,                 -- Previous state root
    timestamp       INTEGER NOT NULL,     -- Unix timestamp (ms)
    device_id       BLOB NOT NULL,        -- 16 bytes (BLAKE3-16)
    signature       BLOB NOT NULL,        -- Ed25519 signature (64 bytes)
    created_at      INTEGER NOT NULL DEFAULT (strftime('%s', 'now') * 1000)
);

CREATE INDEX idx_state_roots_hash ON state_roots(hash);
CREATE INDEX idx_state_roots_parent ON state_roots(parent_hash);
```

#### Full-Text Search

```sql
-- FTS5 virtual table for lexical search
CREATE VIRTUAL TABLE fts_chunks USING fts5(
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.rowid, old.text);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES('delete', old.rowid, old.text);
    INSERT INTO fts_chunks(rowid, text) VALUES (new.rowid, new.text);
END;
```

### 3. HNSW Index

#### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| M | 16 | Number of connections per node |
| ef_construction | 200 | Search quality during construction |
| ef_search | 50 | Search quality during queries |
| max_elements | 100,000 | Initial capacity (auto-resizes) |
| distance | Cosine | Distance metric |

#### Initialization

#### Disk-Based Persistence

To avoid OOM errors on mobile and expensive rebuilds, the HNSW index MUST be backed by disk (e.g., Memory-Mapped Files or DiskANN).

```
function load_or_build_index(embeddings: Vec<Embedding>, state_root: Hash) -> HNSWIndex:
    // 1. Check if on-disk index exists and is valid
    if fs::exists("index.bin") and fs::exists("index.meta"):
        meta = read_meta("index.meta")
        if meta.checkpoint_root == state_root:
             // Fast Path: Map file into memory (zero-copy)
             return HNSWIndex::mmap("index.bin")

    // 2. Slow Path: Rebuild from source
    // Run in background to avoid blocking UI
    return rebuild_index_background(embeddings, state_root)
```

**Constraint**: The index is a **cache**. If it is corrupted or outdated, it is deleted and rebuilt from the SQLite source of truth.

#### Vector Serialization

Embeddings are stored as i16 in SQLite and loaded into HNSW:

```
function serialize_vector(vector: Vec<i16>) -> Vec<u8>:
    return vector.flat_map(|v| v.to_le_bytes())

function deserialize_vector(bytes: Vec<u8>) -> Vec<i16>:
    return Vec<i16>::from_le_bytes(bytes)
```

### 4. GraphStore Interface

```
interface GraphStore:
    // Document operations
    fn insert_document(doc: Document) -> Result<()>
    fn get_document(id: ID) -> Result<Option<Document>>
    fn get_document_by_path(path: String) -> Result<Option<Document>>
    fn delete_document(id: ID) -> Result<()>
    fn all_documents() -> Result<Vec<Document>>

    // Chunk operations
    fn insert_chunk(chunk: Chunk) -> Result<()>
    fn get_chunk(id: ID) -> Result<Option<Chunk>>
    fn chunks_for_document(doc_id: ID) -> Result<Vec<Chunk>>
    fn delete_chunks_for_document(doc_id: ID) -> Result<()>
    fn all_chunks() -> Result<Vec<Chunk>>

    // Embedding operations
    fn insert_embedding(emb: Embedding) -> Result<()>
    fn get_embedding(id: ID) -> Result<Option<Embedding>>
    fn embeddings_for_chunk(chunk_id: ID) -> Result<Vec<Embedding>>
    fn all_embeddings() -> Result<Vec<Embedding>>

    // Vector search
    fn vector_search(query: Vec<i16>, k: usize) -> Result<Vec<(ID, f32)>>

    // Full-text search
    fn fts_search(query: String, limit: usize) -> Result<Vec<(ID, f32)>>

    // Edge operations
    fn insert_edge(edge: Edge) -> Result<()>
    fn edges_from(source_id: ID) -> Result<Vec<Edge>>
    fn edges_to(target_id: ID) -> Result<Vec<Edge>>
    fn all_edges() -> Result<Vec<Edge>>

    // State operations
    fn compute_state_root() -> Result<[u8; 32]>
    fn insert_state_root(root: StateRoot) -> Result<()>
    fn get_latest_state_root() -> Result<Option<StateRoot>>
    fn get_state_root_by_hash(hash: [u8; 32]) -> Result<Option<StateRoot>>

    // Transactions
    fn begin_transaction() -> Result<Transaction>
    fn commit_transaction(tx: Transaction) -> Result<()>
    fn rollback_transaction(tx: Transaction) -> Result<()>

    // Statistics
    fn stats() -> Result<GraphStats>
```

#### GraphStats

```
struct GraphStats {
    document_count: u64,
    chunk_count: u64,
    embedding_count: u64,
    edge_count: u64,
    state_root_count: u64,
    db_size_bytes: u64,
    hnsw_size_bytes: u64,
}
```

### 5. Transaction Semantics

#### ACID Properties

CFS transactions provide:

1. **Atomicity**: All operations in a transaction succeed or all fail
2. **Consistency**: Database invariants are maintained
3. **Isolation**: Concurrent reads see consistent snapshots
4. **Durability**: Committed transactions survive crashes

#### Transaction Workflow

```
function ingest_document(path: String, content: bytes) -> Result<()>:
    tx = graph.begin_transaction()

    try:
        // 1. Parse and chunk document
        doc = create_document(path, content)
        chunks = chunk_document(doc, content)

        // 2. Generate embeddings
        embeddings = embed_chunks(chunks)

        // 3. Insert all entities
        graph.insert_document(doc)
        for chunk in chunks:
            graph.insert_chunk(chunk)
        for emb in embeddings:
            graph.insert_embedding(emb)
            hnsw_index.add(emb.id, emb.vector)

        // 4. Create edges
        for (chunk, emb) in zip(chunks, embeddings):
            graph.insert_edge(Edge::ChunkToEmbedding(chunk.id, emb.id))

        // 5. Commit
        graph.commit_transaction(tx)

    except error:
        graph.rollback_transaction(tx)
        raise error
```

### 6. Index Reconstruction

The HNSW index is rebuilt on startup from SQLite data:

```
function open_graph_store(db_path: String, index_path: String) -> GraphStore:
    // 1. Open SQLite database
    db = SQLite::open(db_path)
    current_root = db.get_latest_state_root()

    // 2. Try to load persistent index
    hnsw = load_or_build_index(db, index_path, current_root)

    // 3. Return initialized store
    return GraphStore { db, hnsw }
```

#### Incremental Updates

After initial construction, the HNSW index is updated incrementally:

```
function add_embedding(emb: Embedding):
    // 1. Insert into SQLite
    db.insert_embedding(emb)

    // 2. Add to HNSW
    vector = deserialize_vector(emb.vector)
    hnsw.add(emb.id, vector)

function remove_embedding(id: ID):
    // 1. Mark as deleted in HNSW (lazy deletion)
    hnsw.mark_deleted(id)

    // 2. Remove from SQLite
    db.delete_embedding(id)
```

### 7. Cascade Deletion

When a document is deleted, all related entities are automatically removed:

```sql
-- Cascades are defined in foreign key constraints:
-- chunks.document_id REFERENCES documents(id) ON DELETE CASCADE
-- embeddings.chunk_id REFERENCES chunks(id) ON DELETE CASCADE

-- Application must also update HNSW index:
function delete_document(doc_id: ID):
    // 1. Get all embeddings that will be deleted
    chunk_ids = db.query(
        "SELECT id FROM chunks WHERE document_id = ?",
        doc_id
    )
    emb_ids = db.query(
        "SELECT id FROM embeddings WHERE chunk_id IN (?)",
        chunk_ids
    )

    // 2. Mark embeddings as deleted in HNSW
    for emb_id in emb_ids:
        hnsw.mark_deleted(emb_id)

    // 3. Delete document (cascades to chunks and embeddings)
    db.execute("DELETE FROM documents WHERE id = ?", doc_id)
```

### 8. Database Migrations

CFS uses versioned migrations for schema changes:

```sql
-- Migration metadata table
CREATE TABLE schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  INTEGER NOT NULL
);

-- Example migration (v2: add tags to documents)
-- Migration 002_add_document_tags.sql
ALTER TABLE documents ADD COLUMN tags TEXT DEFAULT '[]';
```

#### Migration Workflow

```
function migrate_database(db: SQLite):
    current = db.query("SELECT MAX(version) FROM schema_version")

    migrations = load_migrations("migrations/")

    for migration in migrations.where(|m| m.version > current):
        db.execute(migration.sql)
        db.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            migration.version,
            now()
        )
```

## Desired Properties

### 1. Persistence

**Property**: All committed data survives process restarts.

**Mechanism**: SQLite with WAL mode for durability.

### 2. Consistency

**Property**: The database is always in a valid state.

**Mechanism**:
- Foreign key constraints
- Unique constraints on IDs
- Transactional writes

### 3. Isolation

**Property**: Concurrent reads see consistent snapshots.

**Mechanism**: SQLite's MVCC (Multi-Version Concurrency Control).

### 4. Query Efficiency

**Property**: Similarity search completes in sub-linear time.

**Mechanism**: HNSW provides O(log N) query complexity.

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Insert document | O(C × E) | C = chunks, E = embedding time |
| Vector search | O(log N) | N = total embeddings |
| FTS search | O(log N) | BM25-based ranking |
| Delete document | O(C) | C = chunks in document |
| State root computation | O(N log N) | N = total entities |

## Platform Considerations

### Desktop (SQLite 3.x)

- Full WAL mode support
- FTS5 extension enabled
- No special considerations

### Mobile (SQLite 3.x via SQLite.swift / rusqlite)

- WAL mode may have limitations on some devices
- Smaller default page size for memory efficiency
- Background indexing to avoid UI blocking

## Test Vectors

### Schema Validation

```sql
-- Verify schema matches specification
SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name;
```

### HNSW Consistency

```
// Verify HNSW matches SQLite
sqlite_count = db.query("SELECT COUNT(*) FROM embeddings")
hnsw_count = hnsw.len()
assert(sqlite_count == hnsw_count)
```

## References

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [SQLite FTS5 Extension](https://www.sqlite.org/fts5.html)
- [HNSW Paper](https://arxiv.org/abs/1603.09320)
- [Hierarchical Navigable Small World Graphs](https://github.com/nmslib/hnswlib)
