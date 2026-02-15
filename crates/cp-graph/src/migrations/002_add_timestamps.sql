-- Migration 002: Add timestamps to all entity tables
-- Per CP-002: Track creation and update times for auditing

-- Add timestamps to documents
ALTER TABLE documents ADD COLUMN created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));
ALTER TABLE documents ADD COLUMN updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));

-- Add timestamps to chunks
ALTER TABLE chunks ADD COLUMN created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));
ALTER TABLE chunks ADD COLUMN updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));

-- Add timestamps to embeddings
ALTER TABLE embeddings ADD COLUMN created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));
ALTER TABLE embeddings ADD COLUMN updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));

-- Add timestamps to edges
ALTER TABLE edges ADD COLUMN created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));
ALTER TABLE edges ADD COLUMN updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'));

-- Create triggers to auto-update updated_at on UPDATE
CREATE TRIGGER IF NOT EXISTS documents_update_timestamp
    AFTER UPDATE ON documents
BEGIN
    UPDATE documents SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS chunks_update_timestamp
    AFTER UPDATE ON chunks
BEGIN
    UPDATE chunks SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS embeddings_update_timestamp
    AFTER UPDATE ON embeddings
BEGIN
    UPDATE embeddings SET updated_at = strftime('%s', 'now') WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS edges_update_timestamp
    AFTER UPDATE ON edges
BEGIN
    UPDATE edges SET updated_at = strftime('%s', 'now')
    WHERE source = NEW.source AND target = NEW.target AND kind = NEW.kind;
END;
