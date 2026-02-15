//! Chunk node representing a text segment from a document
//!
//! Per CP-011: Chunks use byte offsets and lengths for precise positioning.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A chunk of text extracted from a document
///
/// Documents are split into overlapping chunks for embedding.
/// Each chunk tracks its position within the source document.
///
/// Per CP-011: Uses byte-based offsets (not character-based) for accurate
/// slicing back to original document content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk (BLAKE3-16 of doc_id + sequence + text)
    pub id: Uuid,

    /// Parent document ID
    pub doc_id: Uuid,

    /// The actual text content
    pub text: String,

    /// Byte offset within the source document (u64 for large files)
    pub byte_offset: u64,

    /// Length of this chunk in bytes (u64 for large files)
    pub byte_length: u64,

    /// Sequence number within the document (0-indexed)
    pub sequence: u32,

    /// Hash of the text content for deduplication
    pub text_hash: [u8; 32],
}

impl Chunk {
    /// Create a new chunk with automatic ID generation.
    ///
    /// Per CP-011: ID is BLAKE3-16(doc_id || sequence || text)
    pub fn new(doc_id: Uuid, text: String, byte_offset: u64, sequence: u32) -> Self {
        let text_hash = *blake3::hash(text.as_bytes()).as_bytes();

        // Spec CP-011: ID is composite of doc_id + sequence + text
        let mut hasher = blake3::Hasher::new();
        hasher.update(doc_id.as_bytes());
        hasher.update(&sequence.to_le_bytes());
        hasher.update(text.as_bytes());
        let id_hash = hasher.finalize();

        // Take first 16 bytes for UUID
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&id_hash.as_bytes()[0..16]);
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

    // Legacy field accessors for backwards compatibility during migration
    #[doc(hidden)]
    #[deprecated(note = "Use byte_offset instead")]
    pub fn offset(&self) -> u32 {
        self.byte_offset as u32
    }

    #[doc(hidden)]
    #[deprecated(note = "Use byte_length instead")]
    pub fn len(&self) -> u32 {
        self.byte_length as u32
    }

    #[doc(hidden)]
    #[deprecated(note = "Use sequence instead")]
    pub fn seq(&self) -> u32 {
        self.sequence
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
        assert_eq!(chunk.byte_length, 21);
    }

    #[test]
    fn test_chunk_id_determinism() {
        let doc_id = Uuid::nil();
        let text = "Test text";
        let seq = 1;

        let chunk1 = Chunk::new(doc_id, text.to_string(), 0, seq);
        let chunk2 = Chunk::new(doc_id, text.to_string(), 0, seq);

        assert_eq!(chunk1.id, chunk2.id);

        let chunk3 = Chunk::new(doc_id, "Different text".to_string(), 0, seq);
        assert_ne!(chunk1.id, chunk3.id);

        let chunk4 = Chunk::new(doc_id, text.to_string(), 0, seq + 1);
        assert_ne!(chunk1.id, chunk4.id);
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
    fn test_byte_offset_correctness() {
        let doc_id = Uuid::new_v4();
        let full_text = "Hello, world! This is a test document.";
        let chunk_text = "This is a test";
        let byte_offset = full_text.find(chunk_text).unwrap() as u64;

        let chunk = Chunk::new(doc_id, chunk_text.to_string(), byte_offset, 1);

        // Verify: text[byte_offset..byte_offset+byte_length] == chunk.text
        let slice = &full_text[chunk.byte_offset as usize..(chunk.byte_offset + chunk.byte_length) as usize];
        assert_eq!(slice, chunk.text);
    }
}
