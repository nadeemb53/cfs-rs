//! Document node representing an ingested file

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// A document node in the cognitive graph
///
/// Represents a single ingested file with its content hash
/// for change detection and deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for this document
    pub id: Uuid,

    /// Original file path (relative to watched root)
    pub path: PathBuf,

    /// BLAKE3 hash of file contents
    pub hash: [u8; 32],

    /// Last modification time (Unix timestamp)
    pub mtime: i64,

    /// File size in bytes
    pub size: u64,

    /// MIME type (e.g., "application/pdf", "text/markdown")
    pub mime_type: String,
}

impl Document {
    pub fn new(path: PathBuf, content: &[u8], mtime: i64) -> Self {
        let content_hash = blake3::hash(content);
        let mime_type = mime_from_path(&path);
        
        let id = Uuid::new_v5(&crate::namespaces::DOCUMENT, content_hash.as_bytes());

        Self {
            id,
            path,
            hash: *content_hash.as_bytes(),
            mtime,
            size: content.len() as u64,
            mime_type,
        }
    }

    /// Compute Merkle hash from chunks for provable correctness
    pub fn compute_hierarchical_hash(chunk_hashes: &[[u8; 32]]) -> [u8; 32] {
        let mut section_hasher = blake3::Hasher::new();
        for hash in chunk_hashes {
            section_hasher.update(hash);
        }
        let section_hash = section_hasher.finalize();
        
        // Return hash of section hashes (Merkle root)
        let mut doc_hasher = blake3::Hasher::new();
        doc_hasher.update(section_hash.as_bytes());
        *doc_hasher.finalize().as_bytes()
    }

    /// Check if the document content has changed
    pub fn content_changed(&self, new_content: &[u8]) -> bool {
        let new_hash = blake3::hash(new_content);
        self.hash != *new_hash.as_bytes()
    }

    /// Get the document hash as a hex string
    pub fn hash_hex(&self) -> String {
        hex_encode(&self.hash)
    }
}

/// Infer MIME type from file extension
fn mime_from_path(path: &PathBuf) -> String {
    match path.extension().and_then(|e| e.to_str()) {
        Some("md") | Some("markdown") => "text/markdown".to_string(),
        Some("txt") => "text/plain".to_string(),
        Some("pdf") => "application/pdf".to_string(),
        Some("json") => "application/json".to_string(),
        Some("html") | Some("htm") => "text/html".to_string(),
        _ => "application/octet-stream".to_string(),
    }
}

/// Encode bytes as lowercase hex
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let content = b"Hello, CFS!";
        let doc = Document::new(
            PathBuf::from("test.md"),
            content,
            1234567890,
        );

        assert_eq!(doc.path, PathBuf::from("test.md"));
        assert_eq!(doc.size, 11);
        assert_eq!(doc.mime_type, "text/markdown");
        assert!(!doc.hash_hex().is_empty());
    }

    #[test]
    fn test_content_changed() {
        let content = b"Original content";
        let doc = Document::new(PathBuf::from("test.txt"), content, 0);

        assert!(!doc.content_changed(content));
        assert!(doc.content_changed(b"Modified content"));
    }
}
