-- Migration 001: Initial schema
-- Per CP-002: Core tables for documents, chunks, embeddings, edges, and state roots

CREATE TABLE IF NOT EXISTS documents (
    id BLOB PRIMARY KEY,
    path TEXT NOT NULL,
    hash BLOB NOT NULL,
    hierarchical_hash BLOB NOT NULL,
    mtime INTEGER NOT NULL,
    size INTEGER NOT NULL,
    mime_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id BLOB PRIMARY KEY,
    doc_id BLOB NOT NULL,
    text TEXT NOT NULL,
    byte_offset INTEGER NOT NULL,
    byte_length INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    text_hash BLOB NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings (
    id BLOB PRIMARY KEY,
    chunk_id BLOB NOT NULL,
    vector BLOB NOT NULL,
    model_hash BLOB NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS edges (
    source BLOB NOT NULL,
    target BLOB NOT NULL,
    kind INTEGER NOT NULL,
    weight REAL,
    metadata TEXT,
    PRIMARY KEY (source, target, kind)
);

CREATE TABLE IF NOT EXISTS state_roots (
    hash BLOB PRIMARY KEY,
    parent BLOB,
    hlc_wall_ms INTEGER NOT NULL,
    hlc_counter INTEGER NOT NULL,
    hlc_node_id BLOB NOT NULL,
    device_id BLOB NOT NULL,
    signature BLOB NOT NULL,
    seq INTEGER NOT NULL DEFAULT 0
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id ON embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);

-- FTS5 Virtual Table for lexical search
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    id,
    text,
    content='chunks',
    content_rowid='rowid'
);

-- Triggers to keep FTS in sync with chunks table
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO fts_chunks(rowid, id, text) VALUES (new.rowid, new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, id, text) VALUES('delete', old.rowid, old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, id, text) VALUES('delete', old.rowid, old.id, old.text);
    INSERT INTO fts_chunks(rowid, id, text) VALUES (new.rowid, new.id, new.text);
END;
