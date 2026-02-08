//! State root representing a Merkle commitment to cognitive state

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use uuid::Uuid;

/// A state root representing a Merkle commitment to the entire cognitive state
///
/// Each update to the knowledge graph produces a new state root, forming
/// a chain of cryptographically verifiable state transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRoot {
    /// BLAKE3 hash of the Merkle tree root
    pub hash: [u8; 32],

    /// Hash of the previous state root (None for genesis)
    pub parent: Option<[u8; 32]>,

    /// Unix timestamp when this root was created
    pub timestamp: i64,

    /// Device ID that produced this state
    pub device_id: Uuid,

    /// Ed25519 signature over (hash || parent || timestamp || device_id)
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],

    /// Sequence number for ordering (increments per device)
    pub seq: u64,
}

impl StateRoot {
    /// Create a new unsigned state root
    pub fn new(
        hash: [u8; 32],
        parent: Option<[u8; 32]>,
        timestamp: i64,
        device_id: Uuid,
        seq: u64,
    ) -> Self {
        Self {
            hash,
            parent,
            timestamp,
            device_id,
            signature: [0u8; 64], // Unsigned
            seq,
        }
    }

    /// Get the bytes to be signed
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32 + 32 + 8 + 16 + 8);
        bytes.extend_from_slice(&self.hash);
        if let Some(parent) = &self.parent {
            bytes.extend_from_slice(parent);
        } else {
            bytes.extend_from_slice(&[0u8; 32]);
        }
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(self.device_id.as_bytes());
        bytes.extend_from_slice(&self.seq.to_le_bytes());
        bytes
    }

    /// Check if this is the genesis (first) state root
    pub fn is_genesis(&self) -> bool {
        self.parent.is_none()
    }

    /// Get the hash as a hex string
    pub fn hash_hex(&self) -> String {
        self.hash.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Get the short hash (first 8 chars) for display
    pub fn short_hash(&self) -> String {
        self.hash_hex()[..8].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_root_creation() {
        let root = StateRoot::new(
            [1u8; 32],
            None,
            1234567890,
            Uuid::new_v4(),
            0,
        );

        assert!(root.is_genesis());
        assert_eq!(root.seq, 0);
    }

    #[test]
    fn test_state_chain() {
        let genesis = StateRoot::new([1u8; 32], None, 1000, Uuid::new_v4(), 0);

        let second = StateRoot::new(
            [2u8; 32],
            Some(genesis.hash),
            2000,
            genesis.device_id,
            1,
        );

        assert!(genesis.is_genesis());
        assert!(!second.is_genesis());
        assert_eq!(second.parent, Some(genesis.hash));
    }

    #[test]
    fn test_signing_bytes_deterministic() {
        let root = StateRoot::new(
            [42u8; 32],
            Some([1u8; 32]),
            1234567890,
            Uuid::from_bytes([0u8; 16]),
            5,
        );

        let bytes1 = root.signing_bytes();
        let bytes2 = root.signing_bytes();

        assert_eq!(bytes1, bytes2);
        assert_eq!(bytes1.len(), 32 + 32 + 8 + 16 + 8);
    }
}
