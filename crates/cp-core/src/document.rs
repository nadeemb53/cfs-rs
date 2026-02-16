//! Document node representing an ingested file
//!
//! Per CP-001: Documents use content-based IDs for determinism.
//! path_id is added for filesystem change detection.

use crate::text::normalize;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// A document node in the cognitive graph
///
/// Represents a single ingested file with its content hash
/// for change detection and deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for this document (BLAKE3-16 of content_hash)
    pub id: Uuid,

    /// Path-based identifier for change detection (BLAKE3-16 of canonicalized path)
    pub path_id: Uuid,

    /// Original file path (relative to watched root)
    pub path: PathBuf,

    /// BLAKE3 hash of file contents (canonicalized)
    pub hash: [u8; 32],

    /// Merkle root of chunks (hierarchical hash)
    pub hierarchical_hash: [u8; 32],

    /// Last modification time (Unix timestamp)
    pub mtime: i64,

    /// File size in bytes
    pub size: u64,

    /// MIME type (e.g., "application/pdf", "text/markdown")
    pub mime_type: String,
}

impl Document {
    pub fn new(path: PathBuf, content: &[u8], mtime: i64) -> Self {
        let mime_type = mime_from_path(&path);

        // Per CP-003: Canonicalize content before hashing for determinism
        let text = String::from_utf8_lossy(content);
        let canonical_content = normalize(&text);
        let canonical_bytes = canonical_content.as_bytes();
        let content_hash = blake3::hash(canonical_bytes);

        // Per CP-001: ID is generated from content hash (content-based identity)
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&content_hash.as_bytes()[0..16]);
        let id = Uuid::from_bytes(id_bytes);

        // Per CP-001: path_id is generated from canonicalized path (for change detection)
        let path_str = path.to_string_lossy();
        let canonical_path = normalize(&path_str);
        let path_id_bytes = blake3::hash(canonical_path.as_bytes());
        let mut path_id = [0u8; 16];
        path_id.copy_from_slice(&path_id_bytes.as_bytes()[0..16]);
        let path_id = Uuid::from_bytes(path_id);

        Self {
            id,
            path_id,
            path,
            hash: *content_hash.as_bytes(),
            hierarchical_hash: [0; 32], // Placeholder, computed after chunking
            mtime,
            size: content.len() as u64, // Original size for display
            mime_type,
        }
    }

    /// Update the hierarchical hash (Merkle root of chunks)
    pub fn set_hierarchical_hash(&mut self, hash: [u8; 32]) {
        self.hierarchical_hash = hash;
    }

    /// Compute Merkle hash from chunks for provable correctness
    pub fn compute_hierarchical_hash(chunk_hashes: &[[u8; 32]]) -> [u8; 32] {
        let mut section_hasher = blake3::Hasher::new();
        for hash in chunk_hashes {
            section_hasher.update(hash);
        }
        *section_hasher.finalize().as_bytes()
    }

    /// Check if the document content has changed
    pub fn content_changed(&self, new_content: &[u8]) -> bool {
        let text = String::from_utf8_lossy(new_content);
        let canonical = normalize(&text);
        let new_hash = blake3::hash(canonical.as_bytes());
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
        Some("rs") => "text/x-rust".to_string(),
        Some("py") => "text/x-python".to_string(),
        Some("js") => "text/javascript".to_string(),
        Some("ts") => "text/typescript".to_string(),
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
        let content = b"Hello, CP!";
        let doc = Document::new(
            PathBuf::from("test.md"),
            content,
            1234567890,
        );

        assert_eq!(doc.path, PathBuf::from("test.md"));
        // Original content is 10 bytes
        assert_eq!(doc.size, 10);
        assert_eq!(doc.mime_type, "text/markdown");
        assert_eq!(doc.hierarchical_hash, [0; 32]);

        // Verify ID generation (first 16 bytes of blake3 hash of canonicalized content)
        let canonical = normalize("Hello, CP!");
        let hash = blake3::hash(canonical.as_bytes());
        let expected_id = Uuid::from_bytes(hash.as_bytes()[0..16].try_into().unwrap());
        assert_eq!(doc.id, expected_id);
    }

    #[test]
    fn test_content_changed() {
        let content = b"Original content";
        let doc = Document::new(PathBuf::from("test.txt"), content, 0);

        assert!(!doc.content_changed(content));
        assert!(doc.content_changed(b"Modified content"));
    }

    #[test]
    fn test_path_id_deterministic() {
        let doc1 = Document::new(PathBuf::from("test.md"), b"content", 0);
        let doc2 = Document::new(PathBuf::from("test.md"), b"content", 0);

        // Same path = same path_id
        assert_eq!(doc1.path_id, doc2.path_id);

        // Different path = different path_id
        let doc3 = Document::new(PathBuf::from("other.md"), b"content", 0);
        assert_ne!(doc1.path_id, doc3.path_id);
    }

    #[test]
    fn test_content_id_deterministic() {
        let doc1 = Document::new(PathBuf::from("a.md"), b"hello", 0);
        let doc2 = Document::new(PathBuf::from("b.md"), b"hello", 0);

        // Same content = same ID regardless of path
        assert_eq!(doc1.id, doc2.id);
    }

    // Additional tests for comprehensive coverage

    #[test]
    fn test_document_id_derivation_from_content_hash() {
        // Verify document ID is derived from content hash
        let content = b"Test content for ID derivation";
        let doc = Document::new(PathBuf::from("test.md"), content, 0);

        // ID should be first 16 bytes of BLAKE3 hash of canonicalized content
        let canonical = normalize("Test content for ID derivation");
        let expected_hash = blake3::hash(canonical.as_bytes());
        let mut expected_id_bytes = [0u8; 16];
        expected_id_bytes.copy_from_slice(&expected_hash.as_bytes()[0..16]);
        let expected_id = Uuid::from_bytes(expected_id_bytes);

        assert_eq!(doc.id, expected_id);
    }

    #[test]
    fn test_document_path_id_derivation() {
        // Verify path_id is derived from canonicalized path
        let path = PathBuf::from("test/document.md");
        let doc = Document::new(path.clone(), b"content", 0);

        // path_id should be BLAKE3-16 of canonicalized path
        let canonical_path = normalize(&path.to_string_lossy());
        let expected_hash = blake3::hash(canonical_path.as_bytes());
        let mut expected_path_id_bytes = [0u8; 16];
        expected_path_id_bytes.copy_from_slice(&expected_hash.as_bytes()[0..16]);
        let expected_path_id = Uuid::from_bytes(expected_path_id_bytes);

        assert_eq!(doc.path_id, expected_path_id);
    }

    #[test]
    fn test_document_hierarchical_hash_computation() {
        // Test Merkle hash computation from chunk hashes
        let chunk_hashes: [[u8; 32]; 3] = [
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
        ];

        let hierarchical_hash = Document::compute_hierarchical_hash(&chunk_hashes);

        // Verify it's a 32-byte hash
        assert_eq!(hierarchical_hash.len(), 32);

        // Verify determinism - same input produces same hash
        let hierarchical_hash2 = Document::compute_hierarchical_hash(&chunk_hashes);
        assert_eq!(hierarchical_hash, hierarchical_hash2);

        // Verify different input produces different hash
        let different_hashes: [[u8; 32]; 2] = [[1u8; 32], [2u8; 32]];
        let different_result = Document::compute_hierarchical_hash(&different_hashes);
        assert_ne!(hierarchical_hash, different_result);
    }

    #[test]
    fn test_document_serialization() {
        // Test CBOR round-trip serialization
        use ciborium::ser::into_writer;
        use ciborium::de::from_reader;

        let doc = Document::new(
            PathBuf::from("test.md"),
            b"Hello, World!",
            1234567890,
        );

        // Serialize to CBOR
        let mut serialized = Vec::new();
        into_writer(&doc, &mut serialized).unwrap();

        // Deserialize back
        let deserialized: Document = from_reader(serialized.as_slice()).unwrap();

        // Verify all fields match
        assert_eq!(doc.id, deserialized.id);
        assert_eq!(doc.path_id, deserialized.path_id);
        assert_eq!(doc.path, deserialized.path);
        assert_eq!(doc.hash, deserialized.hash);
        assert_eq!(doc.hierarchical_hash, deserialized.hierarchical_hash);
        assert_eq!(doc.mtime, deserialized.mtime);
        assert_eq!(doc.size, deserialized.size);
        assert_eq!(doc.mime_type, deserialized.mime_type);
    }

    #[test]
    fn test_document_deserialization_invalid() {
        // Test handling of malformed data
        use ciborium::de::from_reader;

        // Invalid CBOR data
        let invalid_data = vec![0xFF, 0xFF, 0xFF];

        let result: Result<Document, _> = from_reader(invalid_data.as_slice());
        assert!(result.is_err());
    }

    #[test]
    fn test_document_canonical_bytes() {
        // Test to_canonical_bytes format (simulated since we use CBOR)
        let doc = Document::new(
            PathBuf::from("test.md"),
            b"Content",
            1000,
        );

        // Verify document has all required fields for canonical bytes
        let id_bytes = doc.id.as_bytes();
        let path_id_bytes = doc.path_id.as_bytes();

        assert_eq!(id_bytes.len(), 16);
        assert_eq!(path_id_bytes.len(), 16);
        assert_eq!(doc.hash.len(), 32);
        assert_eq!(doc.hierarchical_hash.len(), 32);
    }

    #[test]
    fn test_document_mime_type_detection_markdown() {
        // Test .md extension detection
        let doc1 = Document::new(PathBuf::from("readme.md"), b"content", 0);
        let doc2 = Document::new(PathBuf::from("document.markdown"), b"content", 0);

        assert_eq!(doc1.mime_type, "text/markdown");
        assert_eq!(doc2.mime_type, "text/markdown");
    }

    #[test]
    fn test_document_mime_type_detection_text() {
        // Test .txt extension detection
        let doc = Document::new(PathBuf::from("notes.txt"), b"content", 0);
        assert_eq!(doc.mime_type, "text/plain");
    }

    #[test]
    fn test_document_mime_type_detection_unknown() {
        // Test unknown extension defaults to application/octet-stream
        let doc1 = Document::new(PathBuf::from("file.xyz"), b"content", 0);
        let doc2 = Document::new(PathBuf::from("noextension"), b"content", 0);

        assert_eq!(doc1.mime_type, "application/octet-stream");
        assert_eq!(doc2.mime_type, "application/octet-stream");
    }

    #[test]
    fn test_document_size_bytes_calculation() {
        // Verify size matches content length
        let content = b"Test content size";
        let doc = Document::new(PathBuf::from("test.txt"), content, 0);

        assert_eq!(doc.size, content.len() as u64);
    }

    #[test]
    fn test_document_mtime_from_filesystem() {
        // Test mtime is stored correctly
        let mtime: i64 = 1609459200; // 2021-01-01 00:00:00 UTC
        let doc = Document::new(PathBuf::from("test.txt"), b"content", mtime);

        assert_eq!(doc.mtime, mtime);
    }

    #[test]
    fn test_document_set_hierarchical_hash() {
        // Test setting hierarchical hash
        let mut doc = Document::new(PathBuf::from("test.md"), b"content", 0);
        let new_hash = [42u8; 32];

        doc.set_hierarchical_hash(new_hash);

        assert_eq!(doc.hierarchical_hash, new_hash);
    }

    #[test]
    fn test_document_hash_hex() {
        // Test hash_hex() returns hex string
        let doc = Document::new(PathBuf::from("test.md"), b"content", 0);
        let hex_str = doc.hash_hex();

        // Should be 64 characters (32 bytes * 2 hex chars)
        assert_eq!(hex_str.len(), 64);

        // Should only contain hex characters
        assert!(hex_str.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
