//! Edge types for the semantic graph

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Types of relationships in the cognitive graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EdgeKind {
    /// Document contains a chunk
    DocToChunk = 0,

    /// Chunk has an embedding
    ChunkToEmbedding = 1,

    /// Semantic link between chunks (e.g., similar content)
    ChunkToChunk = 2,

    /// Document has a summary
    DocToSummary = 3,

    /// Document references another document
    DocToDoc = 4,
}

impl EdgeKind {
    /// Convert from u8 representation
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::DocToChunk),
            1 => Some(Self::ChunkToEmbedding),
            2 => Some(Self::ChunkToChunk),
            3 => Some(Self::DocToSummary),
            4 => Some(Self::DocToDoc),
            _ => None,
        }
    }
}

/// An edge in the semantic graph
///
/// Represents a directed relationship between two nodes.
/// Per CFS-001 §2.5: edges connect nodes with typed relationships.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    /// Source node ID
    pub source: Uuid,

    /// Target node ID
    pub target: Uuid,

    /// Type of relationship
    pub kind: EdgeKind,

    /// Optional weight/score (e.g., similarity score for ChunkToChunk)
    /// Per CFS-001: stored as f32 in range [0.0, 1.0]
    pub weight: Option<f32>,

    /// Optional metadata string (e.g., relationship context)
    pub metadata: Option<String>,
}

impl Edge {
    /// Create a new edge
    pub fn new(source: Uuid, target: Uuid, kind: EdgeKind) -> Self {
        Self {
            source,
            target,
            kind,
            weight: None,
            metadata: None,
        }
    }

    /// Create a weighted edge (e.g., for similarity scores)
    pub fn with_weight(source: Uuid, target: Uuid, kind: EdgeKind, weight: f32) -> Self {
        Self {
            source,
            target,
            kind,
            weight: Some(weight),
            metadata: None,
        }
    }

    /// Create a doc-to-chunk edge
    pub fn doc_to_chunk(doc_id: Uuid, chunk_id: Uuid) -> Self {
        Self::new(doc_id, chunk_id, EdgeKind::DocToChunk)
    }

    /// Create a chunk-to-embedding edge
    pub fn chunk_to_embedding(chunk_id: Uuid, embedding_id: Uuid) -> Self {
        Self::new(chunk_id, embedding_id, EdgeKind::ChunkToEmbedding)
    }
}

impl Eq for Edge {}

impl std::hash::Hash for Edge {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
        self.target.hash(state);
        self.kind.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let src = Uuid::from_bytes([1u8; 16]);
        let tgt = Uuid::from_bytes([2u8; 16]);

        let edge = Edge::doc_to_chunk(src, tgt);
        assert_eq!(edge.source, src);
        assert_eq!(edge.target, tgt);
        assert_eq!(edge.kind, EdgeKind::DocToChunk);
        assert!(edge.weight.is_none());
        assert!(edge.metadata.is_none());
    }

    #[test]
    fn test_weighted_edge() {
        let edge = Edge::with_weight(
            Uuid::from_bytes([1u8; 16]),
            Uuid::from_bytes([2u8; 16]),
            EdgeKind::ChunkToChunk,
            0.85,
        );

        assert!((edge.weight.unwrap() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_edge_kind_roundtrip() {
        for kind in [
            EdgeKind::DocToChunk,
            EdgeKind::ChunkToEmbedding,
            EdgeKind::ChunkToChunk,
            EdgeKind::DocToSummary,
            EdgeKind::DocToDoc,
        ] {
            let value = kind as u8;
            assert_eq!(EdgeKind::from_u8(value), Some(kind));
        }
    }

    #[test]
    fn test_edge_with_metadata() {
        let mut edge = Edge::new(
            Uuid::from_bytes([1u8; 16]),
            Uuid::from_bytes([2u8; 16]),
            EdgeKind::ChunkToChunk,
        );
        edge.metadata = Some("related by topic".to_string());
        assert_eq!(edge.metadata.as_deref(), Some("related by topic"));
    }
}
