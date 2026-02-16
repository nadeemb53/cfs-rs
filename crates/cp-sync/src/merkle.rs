//! Merkle tree for cognitive state

use cp_core::Result;

/// Merkle tree for computing state roots
pub struct MerkleTree {
    /// Root hash of the tree
    root: Option<[u8; 32]>,
    /// Leaf hashes
    leaves: Vec<[u8; 32]>,
}

impl MerkleTree {
    /// Create an empty Merkle tree
    pub fn new() -> Self {
        Self {
            root: None,
            leaves: Vec::new(),
        }
    }

    /// Add a leaf to the tree
    pub fn add_leaf(&mut self, data: &[u8]) {
        let hash = blake3::hash(data);
        self.leaves.push(*hash.as_bytes());
        self.root = None; // Invalidate cached root
    }

    /// Compute the root hash
    pub fn root(&mut self) -> [u8; 32] {
        if let Some(root) = self.root {
            return root;
        }

        if self.leaves.is_empty() {
            return [0u8; 32];
        }

        let root = self.compute_root(&self.leaves);
        self.root = Some(root);
        root
    }

    /// Recursively compute root from leaves
    fn compute_root(&self, hashes: &[[u8; 32]]) -> [u8; 32] {
        if hashes.is_empty() {
            return [0u8; 32];
        }
        if hashes.len() == 1 {
            return hashes[0];
        }

        let mut next_level = Vec::with_capacity((hashes.len() + 1) / 2);

        for chunk in hashes.chunks(2) {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&chunk[0]);
            if chunk.len() > 1 {
                hasher.update(&chunk[1]);
            } else {
                // Duplicate last hash if odd number
                hasher.update(&chunk[0]);
            }
            next_level.push(*hasher.finalize().as_bytes());
        }

        self.compute_root(&next_level)
    }

    /// Get number of leaves
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Check if tree is empty
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Clear all leaves
    pub fn clear(&mut self) {
        self.leaves.clear();
        self.root = None;
    }
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute a Merkle root from a set of items
#[allow(dead_code)]
pub fn compute_merkle_root<T, F>(items: &[T], hash_fn: F) -> Result<[u8; 32]>
where
    F: Fn(&T) -> [u8; 32],
{
    let mut tree = MerkleTree::new();
    for item in items {
        let hash = hash_fn(item);
        tree.leaves.push(hash);
    }
    Ok(tree.root())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::{Chunk, CognitiveDiff, Document, Edge, EdgeKind, Embedding, Hlc};
    use uuid::Uuid;

    // Helper to create a simple embedding from f32 vector
    fn create_test_embedding(chunk_id: Uuid) -> Embedding {
        // Create a simple f32 vector for testing
        let vector: Vec<f32> = vec![0.0; 384];
        Embedding::new(chunk_id, &vector, [0u8; 32])
    }

    #[test]
    fn test_empty_tree() {
        let mut tree = MerkleTree::new();
        assert_eq!(tree.root(), [0u8; 32]);
    }

    #[test]
    fn test_single_leaf() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"hello");

        let expected = blake3::hash(b"hello");
        assert_eq!(tree.root(), *expected.as_bytes());
    }

    #[test]
    fn test_two_leaves() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"hello");
        tree.add_leaf(b"world");

        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_deterministic() {
        let mut tree1 = MerkleTree::new();
        tree1.add_leaf(b"a");
        tree1.add_leaf(b"b");
        tree1.add_leaf(b"c");

        let mut tree2 = MerkleTree::new();
        tree2.add_leaf(b"a");
        tree2.add_leaf(b"b");
        tree2.add_leaf(b"c");

        assert_eq!(tree1.root(), tree2.root());
    }

    // Additional comprehensive tests for MerkleTree

    #[test]
    fn test_merkle_three_leaves() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"first");
        tree.add_leaf(b"second");
        tree.add_leaf(b"third");

        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn test_merkle_four_leaves() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"one");
        tree.add_leaf(b"two");
        tree.add_leaf(b"three");
        tree.add_leaf(b"four");

        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
        assert_eq!(tree.len(), 4);
    }

    #[test]
    fn test_merkle_many_leaves() {
        let mut tree = MerkleTree::new();
        for i in 0..100 {
            tree.add_leaf(format!("item{}", i).as_bytes());
        }

        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
        assert_eq!(tree.len(), 100);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_merkle_clear() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"test");
        assert_eq!(tree.len(), 1);

        tree.clear();
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
        assert_eq!(tree.root(), [0u8; 32]);
    }

    #[test]
    fn test_merkle_root_caching() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"data");

        let root1 = tree.root();
        let root2 = tree.root();

        // Should return cached root
        assert_eq!(root1, root2);
    }

    #[test]
    fn test_merkle_root_invalidated_on_new_leaf() {
        let mut tree = MerkleTree::new();
        tree.add_leaf(b"first");

        let root1 = tree.root();

        // Adding new leaf should invalidate cache
        tree.add_leaf(b"second");

        let root2 = tree.root();

        assert_ne!(root1, root2);
    }

    #[test]
    fn test_merkle_different_data_different_root() {
        let mut tree1 = MerkleTree::new();
        tree1.add_leaf(b"hello");
        tree1.add_leaf(b"world");

        let mut tree2 = MerkleTree::new();
        tree2.add_leaf(b"foo");
        tree2.add_leaf(b"bar");

        assert_ne!(tree1.root(), tree2.root());
    }

    #[test]
    fn test_merkle_order_matters() {
        let mut tree1 = MerkleTree::new();
        tree1.add_leaf(b"a");
        tree1.add_leaf(b"b");

        let mut tree2 = MerkleTree::new();
        tree2.add_leaf(b"b");
        tree2.add_leaf(b"a");

        // Different order should produce different root
        assert_ne!(tree1.root(), tree2.root());
    }

    #[test]
    fn test_compute_merkle_root_function() {
        let items = vec![
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
        ];

        let root = compute_merkle_root(&items, |item| *item).unwrap();
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_compute_merkle_root_empty() {
        let items: Vec<[u8; 32]> = vec![];
        let root = compute_merkle_root(&items, |item| *item).unwrap();
        assert_eq!(root, [0u8; 32]);
    }

    #[test]
    fn test_compute_merkle_root_single_item() {
        let items = vec![[42u8; 32]];
        let root = compute_merkle_root(&items, |item| *item).unwrap();
        assert_eq!(root, [42u8; 32]);
    }

    // Tests for diff-related functionality using CognitiveDiff

    #[test]
    fn test_merkle_diff_empty() {
        // Test that an empty diff produces expected results
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        assert!(diff.is_empty());
        assert_eq!(diff.change_count(), 0);
    }

    #[test]
    fn test_merkle_diff_document_added() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let doc = Document::new(
            std::path::PathBuf::from("test.md"),
            b"test content",
            0,
        );
        diff.added_docs.push(doc);

        assert!(!diff.is_empty());
        assert_eq!(diff.change_count(), 1);
        assert_eq!(diff.added_docs.len(), 1);
    }

    #[test]
    fn test_merkle_diff_document_modified() {
        // Document modification is modeled as remove + add
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let old_doc_id = Uuid::new_v4();
        diff.removed_doc_ids.push(old_doc_id);

        let new_doc = Document::new(
            std::path::PathBuf::from("test.md"),
            b"updated content",
            0,
        );
        diff.added_docs.push(new_doc);

        assert!(!diff.is_empty());
        assert_eq!(diff.change_count(), 2);
    }

    #[test]
    fn test_merkle_diff_document_deleted() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let doc_id = Uuid::new_v4();
        diff.removed_doc_ids.push(doc_id);

        assert!(!diff.is_empty());
        assert_eq!(diff.change_count(), 1);
        assert_eq!(diff.removed_doc_ids.len(), 1);
    }

    #[test]
    fn test_merkle_diff_chunk_changes() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let doc_id = Uuid::new_v4();
        let chunk = Chunk::new(
            doc_id,
            "test chunk content".to_string(),
            0,
            0,
        );
        diff.added_chunks.push(chunk);

        let removed_chunk_id = Uuid::new_v4();
        diff.removed_chunk_ids.push(removed_chunk_id);

        assert_eq!(diff.change_count(), 2);
        assert_eq!(diff.added_chunks.len(), 1);
        assert_eq!(diff.removed_chunk_ids.len(), 1);
    }

    #[test]
    fn test_merkle_diff_embedding_changes() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let chunk_id = Uuid::new_v4();
        let embedding = create_test_embedding(chunk_id);
        diff.added_embeddings.push(embedding);

        let removed_embedding_id = Uuid::new_v4();
        diff.removed_embedding_ids.push(removed_embedding_id);

        assert_eq!(diff.change_count(), 2);
    }

    #[test]
    fn test_merkle_diff_edge_changes() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let edge = Edge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeKind::DocToChunk,
        );
        diff.added_edges.push(edge);

        let removed_edge = (
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeKind::ChunkToEmbedding,
        );
        diff.removed_edges.push(removed_edge);

        assert_eq!(diff.change_count(), 2);
    }

    #[test]
    fn test_merkle_diff_multiple_changes() {
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        // Add document
        let doc = Document::new(
            std::path::PathBuf::from("test.md"),
            b"content",
            0,
        );
        diff.added_docs.push(doc);

        // Add chunk
        let chunk = Chunk::new(
            Uuid::new_v4(),
            "content".to_string(),
            0,
            0,
        );
        diff.added_chunks.push(chunk);

        // Add embedding
        let embedding = create_test_embedding(Uuid::new_v4());
        diff.added_embeddings.push(embedding);

        // Add edge
        let edge = Edge::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            EdgeKind::DocToChunk,
        );
        diff.added_edges.push(edge);

        assert_eq!(diff.change_count(), 4);
    }

    #[test]
    fn test_merkle_diff_serialization() {
        use crate::{serialize_diff, deserialize_diff};

        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let serialized = serialize_diff(&diff).unwrap();
        assert!(!serialized.is_empty());

        let deserialized = deserialize_diff(&serialized).unwrap();
        assert_eq!(diff.metadata, deserialized.metadata);
    }

    #[test]
    fn test_merkle_diff_serialization_with_content() {
        use crate::{serialize_diff, deserialize_diff};

        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let doc = Document::new(
            std::path::PathBuf::from("test.md"),
            b"Test content",
            0,
        );
        diff.added_docs.push(doc);

        let serialized = serialize_diff(&diff).unwrap();
        let deserialized = deserialize_diff(&serialized).unwrap();

        assert_eq!(diff.added_docs.len(), deserialized.added_docs.len());
        assert_eq!(diff.added_docs[0].path, deserialized.added_docs[0].path);
    }

    #[test]
    fn test_merkle_diff_estimated_size() {
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let size = diff.estimated_size();
        assert!(size > 0);
    }

    #[test]
    fn test_merkle_diff_apply_order() {
        // Tests that changes are tracked in the correct order
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        // Add in specific order
        for i in 0..5 {
            let doc = Document::new(
                std::path::PathBuf::from(format!("doc{}.md", i)),
                format!("content{}", i).as_bytes(),
                0,
            );
            diff.added_docs.push(doc);
        }

        assert_eq!(diff.added_docs.len(), 5);
        // Verify order is preserved
        for (i, doc) in diff.added_docs.iter().enumerate() {
            assert!(doc.path.to_string_lossy().contains(&format!("{}", i)));
        }
    }

    #[test]
    fn test_merkle_diff_idempotent() {
        // Test that adding the same item twice creates a diff that would
        // contain both additions (idempotency is handled at application time)
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        let doc = Document::new(
            std::path::PathBuf::from("test.md"),
            b"content",
            0,
        );

        // Add same document type twice (in real use, this would be deduplicated at apply time)
        diff.added_docs.push(doc.clone());
        diff.added_docs.push(doc);

        assert_eq!(diff.change_count(), 2);
    }

    #[test]
    fn test_merkle_comparison_using_blake3() {
        // Test using Merkle tree to compare two states
        let mut old_tree = MerkleTree::new();
        old_tree.add_leaf(b"document1");
        old_tree.add_leaf(b"document2");

        let mut new_tree = MerkleTree::new();
        new_tree.add_leaf(b"document1");
        new_tree.add_leaf(b"document2");
        new_tree.add_leaf(b"document3"); // New document

        let old_root = old_tree.root();
        let new_root = new_tree.root();

        // Roots should be different because states are different
        assert_ne!(old_root, new_root);
    }

    #[test]
    fn test_merkle_prove_change_detection() {
        // Test that Merkle tree can detect changes
        let mut tree1 = MerkleTree::new();
        tree1.add_leaf(b"a");
        tree1.add_leaf(b"b");

        let mut tree2 = MerkleTree::new();
        tree2.add_leaf(b"a");
        tree2.add_leaf(b"c"); // Changed from b to c

        assert_ne!(tree1.root(), tree2.root());
    }
}
