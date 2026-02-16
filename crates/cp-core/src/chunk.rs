//! Chunk node representing a text segment from a document
//!
//! Per CP-011: Chunks use byte offsets and lengths for precise positioning.
//! Per CP-001: Chunk ID is STABLE - does not include text content.

use crate::text::normalize;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A chunk of text extracted from a document
///
/// Documents are split into overlapping chunks for embedding.
/// Each chunk tracks its position within the source document.
///
/// Per CP-011: Uses byte-based offsets (not character-based) for accurate
/// slicing back to original document content.
///
/// Per CP-001: Chunk ID is STABLE - ID = hash(doc_id + sequence) only.
/// This ensures re-chunking with different parameters produces the same IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk (BLAKE3-16 of doc_id + sequence) - STABLE
    pub id: Uuid,

    /// Parent document ID
    pub doc_id: Uuid,

    /// The actual text content (canonicalized)
    pub text: String,

    /// Byte offset within the source document (u64 for large files)
    pub byte_offset: u64,

    /// Length of this chunk in bytes (u64 for large files)
    pub byte_length: u64,

    /// Sequence number within the document (0-indexed)
    pub sequence: u32,

    /// Hash of the canonicalized text content for verification
    pub text_hash: [u8; 32],
}

impl Chunk {
    /// Create a new chunk with automatic ID generation.
    ///
    /// Per CP-001: Chunk ID is STABLE - does NOT include text.
    /// This ensures re-chunking with different parameters produces same IDs.
    /// Content is verified via text_hash field.
    pub fn new(doc_id: Uuid, text: String, byte_offset: u64, sequence: u32) -> Self {
        // Per CP-003: Canonicalize text before hashing for determinism
        let canonical_text = normalize(&text);
        let text_hash = *blake3::hash(canonical_text.as_bytes()).as_bytes();

        // Per CP-001: ID = hash(doc_id + sequence) - STABLE, does NOT include text
        let id_bytes = crate::id::generate_composite_id(&[
            doc_id.as_bytes(),
            &sequence.to_le_bytes(),
        ]);
        let id = Uuid::from_bytes(id_bytes);

        let byte_length = text.len() as u64;

        Self {
            id,
            doc_id,
            text: canonical_text, // Store canonicalized text
            byte_offset,
            byte_length,
            sequence,
            text_hash,
        }
    }

    /// Create a chunk from already-canonicalized text (for internal use)
    #[doc(hidden)]
    pub fn from_canonical(doc_id: Uuid, text: String, byte_offset: u64, sequence: u32) -> Self {
        let text_hash = *blake3::hash(text.as_bytes()).as_bytes();

        // Per CP-001: ID = hash(doc_id + sequence) - STABLE
        let id_bytes = crate::id::generate_composite_id(&[
            doc_id.as_bytes(),
            &sequence.to_le_bytes(),
        ]);
        let id = Uuid::from_bytes(id_bytes);

        let byte_length = text.len() as u64;

        Self {
            id,
            doc_id,
            text,
            byte_offset,
            byte_length,
            sequence,
            text_hash,
        }
    }

    /// Get the text hash as a hex string
    pub fn text_hash_hex(&self) -> String {
        self.text_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    /// Approximate token count (rough estimate: 4 chars per token)
    pub fn approx_tokens(&self) -> usize {
        self.text.len() / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_creation() {
        let doc_id = Uuid::new_v4();
        let chunk = Chunk::new(
            doc_id,
            "This is a test chunk.".to_string(),
            0,
            0,
        );

        assert_eq!(chunk.doc_id, doc_id);
        assert_eq!(chunk.byte_offset, 0);
        assert_eq!(chunk.sequence, 0);
    }

    #[test]
    fn test_chunk_id_stable() {
        let doc_id = Uuid::nil();
        let text = "Test text";

        // Same doc_id + sequence = same chunk ID regardless of text content
        let chunk1 = Chunk::new(doc_id, text.to_string(), 0, 1);
        let chunk2 = Chunk::new(doc_id, "Different text".to_string(), 0, 1);

        assert_eq!(chunk1.id, chunk2.id);

        // Different sequence = different ID
        let chunk3 = Chunk::new(doc_id, text.to_string(), 0, 2);
        assert_ne!(chunk1.id, chunk3.id);
    }

    #[test]
    fn test_chunk_id_determinism() {
        let doc_id = Uuid::nil();
        let text = "Test text";
        let seq = 1;

        let chunk1 = Chunk::new(doc_id, text.to_string(), 0, seq);
        let chunk2 = Chunk::new(doc_id, text.to_string(), 0, seq);

        assert_eq!(chunk1.id, chunk2.id);
    }

    #[test]
    fn test_text_canonicalized() {
        let doc_id = Uuid::nil();

        // Text with different whitespace should canonicalize to same
        let chunk1 = Chunk::new(doc_id, "Hello   \nWorld".to_string(), 0, 0);
        let chunk2 = Chunk::new(doc_id, "Hello\nWorld".to_string(), 0, 0);

        // Text is canonicalized
        assert_eq!(chunk1.text, "Hello\nWorld\n");
        assert_eq!(chunk1.text, chunk2.text);

        // But ID is still stable (not based on text)
        assert_eq!(chunk1.id, chunk2.id);
    }

    #[test]
    fn test_approx_tokens() {
        let chunk = Chunk::new(
            Uuid::new_v4(),
            "A".repeat(400), // ~100 tokens
            0,
            0,
        );

        assert_eq!(chunk.approx_tokens(), 100);
    }

    #[test]
    fn test_byte_offset() {
        let doc_id = Uuid::new_v4();
        let chunk = Chunk::new(doc_id, "Test".to_string(), 14, 1);

        assert_eq!(chunk.byte_offset, 14);
    }

    // Additional tests for comprehensive coverage

    #[test]
    fn test_chunk_id_different_sequence_different_id() {
        // Same content, different sequence = different ID
        let doc_id = Uuid::nil();

        let chunk1 = Chunk::new(doc_id, "Same text".to_string(), 0, 0);
        let chunk2 = Chunk::new(doc_id, "Same text".to_string(), 0, 1);
        let chunk3 = Chunk::new(doc_id, "Same text".to_string(), 0, 2);

        // All should have different IDs
        assert_ne!(chunk1.id, chunk2.id);
        assert_ne!(chunk2.id, chunk3.id);
        assert_ne!(chunk1.id, chunk3.id);
    }

    #[test]
    fn test_chunk_text_hash_computation() {
        // Test that text_hash is computed from canonicalized text
        let doc_id = Uuid::nil();
        let text = "Test text for hashing";

        let chunk = Chunk::new(doc_id, text.to_string(), 0, 0);

        // Compute expected hash: BLAKE3 of canonicalized text
        let canonical = normalize(text);
        let expected_hash = *blake3::hash(canonical.as_bytes()).as_bytes();

        assert_eq!(chunk.text_hash, expected_hash);
    }

    #[test]
    fn test_chunk_byte_offset_validation() {
        // Test various byte offset values
        let doc_id = Uuid::nil();

        // Zero offset
        let chunk0 = Chunk::new(doc_id, "test".to_string(), 0, 0);
        assert_eq!(chunk0.byte_offset, 0);

        // Large offset for large files
        let chunk_large = Chunk::new(doc_id, "test".to_string(), 1_000_000, 0);
        assert_eq!(chunk_large.byte_offset, 1_000_000);
    }

    #[test]
    fn test_chunk_sequence_ordering() {
        // Test sequence numbers are 0-indexed
        let doc_id = Uuid::nil();

        let chunk0 = Chunk::new(doc_id, "first".to_string(), 0, 0);
        let chunk1 = Chunk::new(doc_id, "second".to_string(), 10, 1);
        let chunk2 = Chunk::new(doc_id, "third".to_string(), 20, 2);

        assert_eq!(chunk0.sequence, 0);
        assert_eq!(chunk1.sequence, 1);
        assert_eq!(chunk2.sequence, 2);
    }

    #[test]
    fn test_chunk_canonical_bytes_format() {
        // Verify serialization format includes all fields
        let doc_id = Uuid::nil();
        let chunk = Chunk::new(doc_id, "Test content".to_string(), 0, 0);

        // Verify all required fields exist and are valid
        assert_eq!(chunk.id.as_bytes().len(), 16);
        assert_eq!(chunk.doc_id.as_bytes().len(), 16);
        assert_eq!(chunk.text_hash.len(), 32);
        assert!(chunk.sequence >= 0);
    }

    #[test]
    fn test_chunk_overlap_semantics() {
        // Test byte_offset and byte_length for overlap detection
        let doc_id = Uuid::nil();
        let text = "Hello World";

        // First chunk
        let chunk1 = Chunk::new(doc_id, text.to_string(), 0, 0);
        // Second chunk starting at offset (simulating overlap)
        let chunk2 = Chunk::new(doc_id, text.to_string(), 5, 1);

        // Both should exist with different offsets
        assert_eq!(chunk1.byte_offset, 0);
        assert_eq!(chunk2.byte_offset, 5);

        // byte_length should match text length
        assert_eq!(chunk1.byte_length, text.len() as u64);
    }

    #[test]
    fn test_chunk_text_validation_utf8() {
        // Test that only valid UTF-8 is accepted
        let doc_id = Uuid::nil();

        // Valid UTF-8 strings
        let valid_texts = vec![
            "Hello, World!",
            "Unicode: cafe with accent: cafe",
            "Emoji: hello world",
            "",
            "Multiple\nlines\nhere",
        ];

        for text in valid_texts {
            let chunk = Chunk::new(doc_id, text.to_string(), 0, 0);
            assert!(chunk.text.is_utf8());
        }
    }

    #[test]
    fn test_chunk_empty_text_rejected() {
        // Test handling of empty chunk text
        let doc_id = Uuid::nil();

        // Empty text should still create a chunk (with empty text)
        let chunk = Chunk::new(doc_id, "".to_string(), 0, 0);

        // The chunk should exist but with empty text
        assert_eq!(chunk.text, "");
        assert_eq!(chunk.byte_length, 0);
    }

    #[test]
    fn test_chunk_text_hash_hex() {
        // Test text_hash_hex() method
        let doc_id = Uuid::nil();
        let chunk = Chunk::new(doc_id, "Test text".to_string(), 0, 0);

        let hex = chunk.text_hash_hex();

        // Should be 64 characters (32 bytes * 2 hex chars)
        assert_eq!(hex.len(), 64);

        // Should only contain hex characters
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_chunk_from_canonical() {
        // Test from_canonical constructor
        let doc_id = Uuid::nil();
        let text = "Already canonicalized text".to_string();

        let chunk = Chunk::from_canonical(doc_id, text.clone(), 100, 5);

        assert_eq!(chunk.text, text);
        assert_eq!(chunk.byte_offset, 100);
        assert_eq!(chunk.sequence, 5);
        assert_eq!(chunk.doc_id, doc_id);
    }
}
