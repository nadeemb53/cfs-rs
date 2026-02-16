-- Migration 004: Add path_id to documents and embedding_version to embeddings
-- Per CP-001: Document.path_id for change detection, Embedding.embedding_version for re-embedding

-- Add path_id column to documents table
-- path_id is derived from canonicalized path for change detection
-- Existing rows will have path_id populated by app layer on next read
ALTER TABLE documents ADD COLUMN path_id BLOB;

-- Add embedding_version column to embeddings table
-- Used for re-embedding scenarios when model changes
ALTER TABLE embeddings ADD COLUMN embedding_version INTEGER NOT NULL DEFAULT 0;
