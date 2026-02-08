//! Merkle tree for cognitive state

use cfs_core::Result;

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
}
