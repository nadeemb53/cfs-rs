//! Database migration system for CFS
//!
//! Per CFS-002: Provides versioned schema migrations with automatic upgrade.
//! Migrations are embedded in the binary and run in order.

use cfs_core::{CfsError, Result};
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
    .map_err(|e| CfsError::Database(format!("Failed to create schema_version table: {}", e)))?;

    // Get current version
    let current_version: u32 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_version",
            [],
            |row| row.get(0),
        )
        .map_err(|e| CfsError::Database(format!("Failed to get schema version: {}", e)))?;

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
                .map_err(|e| CfsError::Database(format!("Failed to start transaction: {}", e)))?;

            tx.execute_batch(migration.sql)
                .map_err(|e| {
                    CfsError::Database(format!(
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
                CfsError::Database(format!("Failed to record migration: {}", e))
            })?;

            tx.commit()
                .map_err(|e| CfsError::Database(format!("Failed to commit migration: {}", e)))?;

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
        .map_err(|e| CfsError::Database(e.to_string()))?;

    if !table_exists {
        return Ok(0);
    }

    conn.query_row(
        "SELECT COALESCE(MAX(version), 0) FROM schema_version",
        [],
        |row| row.get(0),
    )
    .map_err(|e| CfsError::Database(e.to_string()))
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
        assert_eq!(version, 3);
    }

    #[test]
    fn test_schema_version_tracking() {
        let conn = Connection::open_in_memory().unwrap();

        assert_eq!(get_schema_version(&conn).unwrap(), 0);
        assert!(needs_migration(&conn).unwrap());

        run_migrations(&conn).unwrap();

        assert_eq!(get_schema_version(&conn).unwrap(), 3);
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
}
