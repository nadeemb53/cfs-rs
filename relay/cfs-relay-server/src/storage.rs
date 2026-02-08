//! Storage backend for the relay server

use rusqlite::{params, Connection};
use std::sync::Mutex;

use crate::{RootInfo, UploadRequest};

/// SQLite-based storage for encrypted diffs
/// Uses a Mutex to ensure thread safety with Axum's async handlers
pub struct Storage {
    conn: Mutex<Connection>,
}

// Safety: We use Mutex to ensure only one thread accesses Connection at a time
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    /// Open or create storage at the given path
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = if path == ":memory:" {
            Connection::open_in_memory()?
        } else {
            Connection::open(path)?
        };

        conn.execute(
            "CREATE TABLE IF NOT EXISTS diffs (
                root TEXT PRIMARY KEY,
                ciphertext TEXT NOT NULL,
                nonce TEXT NOT NULL,
                signature TEXT NOT NULL,
                public_key TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                size INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Store an encrypted diff
    pub fn store(
        &self,
        root: &str,
        ciphertext: &str,
        nonce: &str,
        signature: &str,
        public_key: &str,
    ) -> Result<(), rusqlite::Error> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let size = ciphertext.len();

        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO diffs (root, ciphertext, nonce, signature, public_key, timestamp, size)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![root, ciphertext, nonce, signature, public_key, timestamp, size],
        )?;

        Ok(())
    }

    /// List all stored roots
    pub fn list_roots(&self) -> Result<Vec<RootInfo>, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        let mut stmt =
            conn.prepare("SELECT root, timestamp, size FROM diffs ORDER BY timestamp ASC")?;

        let roots = stmt
            .query_map([], |row| {
                Ok(RootInfo {
                    root: row.get(0)?,
                    timestamp: row.get(1)?,
                    size: row.get(2)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(roots)
    }

    /// Get a specific diff
    pub fn get(&self, root: &str) -> Result<UploadRequest, rusqlite::Error> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT ciphertext, nonce, signature, public_key FROM diffs WHERE root = ?1",
            params![root],
            |row| {
                Ok(UploadRequest {
                    ciphertext: row.get(0)?,
                    nonce: row.get(1)?,
                    signature: row.get(2)?,
                    public_key: row.get(3)?,
                    root: root.to_string(),
                })
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_roundtrip() {
        let storage = Storage::open(":memory:").unwrap();

        storage
            .store(
                "abc123",
                "encrypted_data",
                "nonce123",
                "sig123",
                "pubkey123",
            )
            .unwrap();

        let roots = storage.list_roots().unwrap();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].root, "abc123");

        let diff = storage.get("abc123").unwrap();
        assert_eq!(diff.ciphertext, "encrypted_data");
    }
}
