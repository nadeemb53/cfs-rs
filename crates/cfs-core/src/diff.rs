//! Cognitive diff representing changes between state roots
//!
//! Per CFS-001 §2.7: A diff captures all changes between two state roots
//! and is the fundamental unit of synchronization.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::hlc::Hlc;
use crate::{Chunk, Document, Edge, EdgeKind, Embedding};

/// Metadata about a cognitive diff
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffMetadata {
    /// Hash of the previous state root
    pub prev_root: [u8; 32],

    /// Hash of the new state root after applying this diff
    pub new_root: [u8; 32],

    /// HLC timestamp when this diff was created
    pub hlc: Hlc,

    /// Device ID that produced this diff
    pub device_id: Uuid,

    /// Sequence number for ordering
    pub seq: u64,
}

/// A cognitive diff containing all changes between two state roots.
///
/// Per CFS-001 §2.7: devices exchange diffs rather than full state
/// to minimize bandwidth and computation.
///
/// Note: there is no `updated_docs` — updates are modeled as
/// remove + add (per spec pattern).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveDiff {
    /// Documents added in this diff
    pub added_docs: Vec<Document>,

    /// IDs of documents removed
    pub removed_doc_ids: Vec<Uuid>,

    /// Chunks added
    pub added_chunks: Vec<Chunk>,

    /// IDs of chunks removed
    pub removed_chunk_ids: Vec<Uuid>,

    /// Embeddings added
    pub added_embeddings: Vec<Embedding>,

    /// IDs of embeddings removed
    pub removed_embedding_ids: Vec<Uuid>,

    /// Edges added
    pub added_edges: Vec<Edge>,

    /// Edges removed — identified by (source, target, kind) triple
    pub removed_edges: Vec<(Uuid, Uuid, EdgeKind)>,

    /// Diff metadata
    pub metadata: DiffMetadata,
}

impl CognitiveDiff {
    /// Create an empty diff (no changes)
    pub fn empty(prev_root: [u8; 32], device_id: Uuid, seq: u64, hlc: Hlc) -> Self {
        Self {
            added_docs: Vec::new(),
            removed_doc_ids: Vec::new(),
            added_chunks: Vec::new(),
            removed_chunk_ids: Vec::new(),
            added_embeddings: Vec::new(),
            removed_embedding_ids: Vec::new(),
            added_edges: Vec::new(),
            removed_edges: Vec::new(),
            metadata: DiffMetadata {
                prev_root,
                new_root: [0u8; 32], // Computed after application
                hlc,
                device_id,
                seq,
            },
        }
    }

    /// Check if the diff is empty (no changes)
    pub fn is_empty(&self) -> bool {
        self.added_docs.is_empty()
            && self.removed_doc_ids.is_empty()
            && self.added_chunks.is_empty()
            && self.removed_chunk_ids.is_empty()
            && self.added_embeddings.is_empty()
            && self.removed_embedding_ids.is_empty()
            && self.added_edges.is_empty()
            && self.removed_edges.is_empty()
    }

    /// Count total number of changes in this diff
    pub fn change_count(&self) -> usize {
        self.added_docs.len()
            + self.removed_doc_ids.len()
            + self.added_chunks.len()
            + self.removed_chunk_ids.len()
            + self.added_embeddings.len()
            + self.removed_embedding_ids.len()
            + self.added_edges.len()
            + self.removed_edges.len()
    }

    /// Estimate serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        const DOC_SIZE: usize = 200;
        const CHUNK_SIZE: usize = 1000;
        const EMBEDDING_SIZE: usize = 800; // 384 dims * 2 bytes
        const EDGE_SIZE: usize = 50;
        const ID_SIZE: usize = 16;

        self.added_docs.len() * DOC_SIZE
            + self.removed_doc_ids.len() * ID_SIZE
            + self.added_chunks.len() * CHUNK_SIZE
            + self.removed_chunk_ids.len() * ID_SIZE
            + self.added_embeddings.len() * EMBEDDING_SIZE
            + self.removed_embedding_ids.len() * ID_SIZE
            + self.added_edges.len() * EDGE_SIZE
            + self.removed_edges.len() * (ID_SIZE * 2 + 1)
            + 200 // metadata overhead
    }
}

impl PartialEq for CognitiveDiff {
    fn eq(&self, other: &Self) -> bool {
        self.metadata == other.metadata
    }
}

impl Eq for CognitiveDiff {}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_hlc() -> Hlc {
        Hlc::new(1000, [1u8; 16])
    }

    #[test]
    fn test_empty_diff() {
        let diff = CognitiveDiff::empty([0u8; 32], Uuid::from_bytes([1u8; 16]), 0, test_hlc());
        assert!(diff.is_empty());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_diff_with_changes() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::from_bytes([1u8; 16]),
            0,
            test_hlc(),
        );

        diff.added_docs.push(Document::new(
            std::path::PathBuf::from("test.md"),
            b"content",
            0,
        ));

        assert!(!diff.is_empty());
        assert_eq!(diff.change_count(), 1);
    }

    #[test]
    fn test_diff_has_hlc() {
        let hlc = Hlc::new(12345, [7u8; 16]);
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::from_bytes([1u8; 16]),
            0,
            hlc.clone(),
        );
        assert_eq!(diff.metadata.hlc, hlc);
    }

    #[test]
    fn test_diff_removed_edges_has_kind() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::from_bytes([1u8; 16]),
            0,
            test_hlc(),
        );

        diff.removed_edges.push((
            Uuid::from_bytes([2u8; 16]),
            Uuid::from_bytes([3u8; 16]),
            EdgeKind::DocToChunk,
        ));

        assert_eq!(diff.change_count(), 1);
        assert_eq!(diff.removed_edges[0].2, EdgeKind::DocToChunk);
    }
}
