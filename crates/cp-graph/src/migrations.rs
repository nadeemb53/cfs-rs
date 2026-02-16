//! Database migration system for CP
//!
//! Per CP-002: Provides versioned schema migrations with automatic upgrade.
//! Migrations are embedded in the binary and run in order.

use cp_core::{CPError, Result};
use rusqlite::Connection;
use tracing::info;

/// Migration definition
struct Migration {
    version: u32,
    name: &'static str,
    sql: &'static str,
}

/// All migrations in order
const MIGRATIONS: &[Migration] = &[
    Migration {
        version: 1,
        name: "initial_schema",
        sql: include_str!("migrations/001_initial.sql"),
    },
    Migration {
        version: 2,
        name: "add_timestamps",
        sql: include_str!("migrations/002_add_timestamps.sql"),
    },
    Migration {
        version: 3,
        name: "add_l2_norm",
        sql: include_str!("migrations/003_add_l2_norm.sql"),
    },
    Migration {
        version: 4,
        name: "add_path_id_and_embedding_version",
        sql: include_str!("migrations/004_add_path_id_and_embedding_version.sql"),
    },
];

/// Run all pending migrations on the database
pub fn run_migrations(conn: &Connection) -> Result<()> {
    // Create schema_version table if it doesn't exist
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
        );
        "#,
    )
    .map_err(|e| CPError::Database(format!("Failed to create schema_version table: {}", e)))?;

    // Get current version
    let current_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .map_err(|e| CPError::Database(format!("Failed to get schema version: {}", e)))?;

    info!("Current schema version: {}", current_version);

    // Run pending migrations
    for migration in MIGRATIONS {
        if migration.version > current_version {
            info!(
                "Running migration {}: {}",
                migration.version, migration.name
            );

            // Run migration in a transaction
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| CPError::Database(format!("Failed to start transaction: {}", e)))?;

            tx.execute_batch(migration.sql)
                .map_err(|e| {
                    CPError::Database(format!(
                        "Migration {} ({}) failed: {}",
                        migration.version, migration.name, e
                    ))
                })?;

            // Record migration
            tx.execute(
                "INSERT INTO schema_version (version, name) VALUES (?1, ?2)",
                rusqlite::params![migration.version, migration.name],
            )
            .map_err(|e| {
                CPError::Database(format!("Failed to record migration: {}", e))
            })?;

            tx.commit()
                .map_err(|e| CPError::Database(format!("Failed to commit migration: {}", e)))?;

            info!("Migration {} complete", migration.version);
        }
    }

    info!("All migrations complete");
    Ok(())
}

/// Get the current schema version
pub fn get_schema_version(conn: &Connection) -> Result<u32> {
    // Check if schema_version table exists
    let table_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')",
            [],
            |row| row.get(0),
        )
        .map_err(|e| CPError::Database(e.to_string()))?;

    if !table_exists {
        return Ok(0);
    }

    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .map_err(|e| CPError::Database(e.to_string()))
}

/// Check if the database needs migration
pub fn needs_migration(conn: &Connection) -> Result<bool> {
    let current = get_schema_version(conn)?;
    let latest = MIGRATIONS.last().map(|m| m.version).unwrap_or(0);
    Ok(current < latest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migrations_run_idempotent() {
        let conn = Connection::open_in_memory().unwrap();

        // Run migrations twice - should not fail
        run_migrations(&conn).unwrap();
        run_migrations(&conn).unwrap();

        let version = get_schema_version(&conn).unwrap();
        assert_eq!(version, 4);
    }

    #[test]
    fn test_schema_version_tracking() {
        let conn = Connection::open_in_memory().unwrap();

        assert_eq!(get_schema_version(&conn).unwrap(), 0);
        assert!(needs_migration(&conn).unwrap());

        run_migrations(&conn).unwrap();

        assert_eq!(get_schema_version(&conn).unwrap(), 4);
        assert!(!needs_migration(&conn).unwrap());
    }

    #[test]
    fn test_timestamps_exist() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Check that timestamp columns exist
        let has_created_at: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'created_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_created_at);

        let has_updated_at: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'updated_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_updated_at);
    }

    #[test]
    fn test_l2_norm_column_exists() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        let has_l2_norm: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('embeddings') WHERE name = 'l2_norm'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_l2_norm);
    }

    // ========== Additional Migration Tests ==========

    #[test]
    fn test_migration_runner_initial_schema() {
        let conn = Connection::open_in_memory().unwrap();

        // Run migrations on fresh database
        run_migrations(&conn).unwrap();

        // Verify all tables exist
        let doc_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='documents'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(doc_count, 1);

        let chunk_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='chunks'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(chunk_count, 1);

        let emb_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='embeddings'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(emb_count, 1);

        let edge_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='edges'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(edge_count, 1);

        let state_root_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='state_roots'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(state_root_count, 1);
    }

    #[test]
    fn test_migration_already_applied() {
        let conn = Connection::open_in_memory().unwrap();

        // Run migrations first time
        run_migrations(&conn).unwrap();
        let version1 = get_schema_version(&conn).unwrap();
        assert_eq!(version1, 4);

        // Run again - should skip already applied migrations
        run_migrations(&conn).unwrap();
        let version2 = get_schema_version(&conn).unwrap();
        assert_eq!(version2, 4);
    }

    #[test]
    fn test_migration_001_documents_table() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Check documents table schema
        let has_id: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_id);

        let has_path: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'path'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_path);

        let has_hash: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'hash'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_hash);

        let has_mtime: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'mtime'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_mtime);

        let has_size: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'size'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_size);

        let has_mime_type: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'mime_type'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(has_mime_type);
    }

    #[test]
    fn test_migration_002_timestamps() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Verify created_at exists on documents
        let created_at_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'created_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(created_at_exists);

        // Verify updated_at exists on documents
        let updated_at_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'updated_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(updated_at_exists);

        // Verify created_at exists on chunks
        let chunk_created_at: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('chunks') WHERE name = 'created_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(chunk_created_at);

        // Verify created_at exists on embeddings
        let emb_created_at: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('embeddings') WHERE name = 'created_at'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(emb_created_at);
    }

    #[test]
    fn test_migration_003_l2_norm() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Verify l2_norm column exists on embeddings
        let l2_norm_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('embeddings') WHERE name = 'l2_norm'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(l2_norm_exists);
    }

    #[test]
    fn test_migration_004_path_id_embedding_version() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Verify path_id column exists on documents
        let path_id_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('documents') WHERE name = 'path_id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(path_id_exists);

        // Verify embedding_version column exists on embeddings
        let emb_version_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_table_info('embeddings') WHERE name = 'embedding_version'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(emb_version_exists);
    }

    #[test]
    fn test_migration_foreign_keys() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON", []).unwrap();

        // Insert a document
        conn.execute(
            "INSERT INTO documents (id, path, hash, hierarchical_hash, mtime, size, mime_type) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                uuid::Uuid::new_v4().as_bytes(),
                "test.md",
                [0u8; 32].as_slice(),
                [0u8; 32].as_slice(),
                0i64,
                0i64,
                "text/markdown"
            ],
        ).unwrap();

        // Verify chunks table has foreign key to documents
        let fk_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_foreign_key_list('chunks')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(fk_exists);

        // Verify embeddings table has foreign key to chunks
        let emb_fk_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM pragma_foreign_key_list('embeddings')",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(emb_fk_exists);
    }

    #[test]
    fn test_migration_fts_triggers() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Verify FTS virtual table exists
        let fts_exists: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='fts_chunks'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(fts_exists);

        // Verify insert trigger exists
        let ai_trigger: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='trigger' AND name='chunks_ai'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(ai_trigger);

        // Verify delete trigger exists
        let ad_trigger: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='trigger' AND name='chunks_ad'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(ad_trigger);

        // Verify update trigger exists
        let au_trigger: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='trigger' AND name='chunks_au'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(au_trigger);
    }

    #[test]
    fn test_migration_fts_content_sync() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Insert a document
        let doc_id = uuid::Uuid::new_v4();
        conn.execute(
            "INSERT INTO documents (id, path, hash, hierarchical_hash, mtime, size, mime_type) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                doc_id.as_bytes(),
                "test.md",
                [0u8; 32].as_slice(),
                [0u8; 32].as_slice(),
                0i64,
                0i64,
                "text/markdown"
            ],
        ).unwrap();

        // Insert a chunk
        let chunk_id = uuid::Uuid::new_v4();
        conn.execute(
            "INSERT INTO chunks (id, doc_id, text, byte_offset, byte_length, sequence, text_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                chunk_id.as_bytes(),
                doc_id.as_bytes(),
                "test content for search",
                0i64,
                0i64,
                0u32,
                [0u8; 32].as_slice()
            ],
        ).unwrap();

        // Verify FTS table has the content
        let fts_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH 'test'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(fts_count > 0);
    }

    #[test]
    fn test_migration_indexes() {
        let conn = Connection::open_in_memory().unwrap();
        run_migrations(&conn).unwrap();

        // Check for idx_chunks_doc_id
        let idx1: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_chunks_doc_id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(idx1);

        // Check for idx_embeddings_chunk_id
        let idx2: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_embeddings_chunk_id'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(idx2);

        // Check for idx_edges_source
        let idx3: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_edges_source'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(idx3);

        // Check for idx_edges_target
        let idx4: bool = conn
            .query_row(
                "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_edges_target'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(idx4);
    }

    #[test]
    fn test_schema_version_table_structure() {
        let conn = Connection::open_in_memory().unwrap();

        // Create the schema_version table manually to test structure
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
            );
            "#,
        ).unwrap();

        // Insert a test migration record
        conn.execute(
            "INSERT INTO schema_version (version, name) VALUES (1, 'test_migration')",
            [],
        ).unwrap();

        // Verify it was recorded
        let version: i64 = conn
            .query_row("SELECT version FROM schema_version WHERE name = 'test_migration'", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 1);

        let name: String = conn
            .query_row("SELECT name FROM schema_version WHERE version = 1", [], |row| row.get(0))
            .unwrap();
        assert_eq!(name, "test_migration");
    }
}
