-- Migration 003: Add l2_norm column to embeddings
-- Per CFS-001: Precomputed L2 norm for efficient similarity computation

ALTER TABLE embeddings ADD COLUMN l2_norm REAL NOT NULL DEFAULT 0.0;

-- Create index for potential norm-based filtering
CREATE INDEX IF NOT EXISTS idx_embeddings_l2_norm ON embeddings(l2_norm);
