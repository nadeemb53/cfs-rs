//! Chunk node representing a text segment from a document

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A chunk of text extracted from a document
///
/// Documents are split into overlapping chunks for embedding.
/// Each chunk tracks its position within the source document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    /// Unique identifier for this chunk
    pub id: Uuid,

    /// Parent document ID
    pub doc_id: Uuid,

    /// The actual text content
    pub text: String,

    /// Byte offset within the source document
    pub offset: u32,

    /// Length of this chunk in bytes
    pub len: u32,

    /// Sequence number within the document (0-indexed)
    pub seq: u32,

    /// Hash of the text content for deduplication
    pub text_hash: [u8; 32],
}

impl Chunk {
    pub fn new(doc_id: Uuid, text: String, offset: u32, seq: u32) -> Self {
        let text_hash = *blake3::hash(text.as_bytes()).as_bytes();
        
        let id = Uuid::new_v5(&crate::namespaces::CHUNK, &text_hash);

        let len = text.len() as u32;

        Self {
            id,
            doc_id,
            text,
            offset,
            len,
            seq,
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
        assert_eq!(chunk.offset, 0);
        assert_eq!(chunk.seq, 0);
        assert_eq!(chunk.len, 21);
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
}
