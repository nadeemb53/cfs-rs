//! State root representing a Merkle commitment to cognitive state
//!
//! Per CP-002 §2.6: Each update to the knowledge graph produces a new state root,
//! forming a chain of cryptographically verifiable state transitions.

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use uuid::Uuid;

use crate::hlc::Hlc;

/// A state root representing a Merkle commitment to the entire cognitive state.
///
/// Each update to the knowledge graph produces a new state root, forming
/// a chain of cryptographically verifiable state transitions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRoot {
    /// BLAKE3 hash of the Merkle tree root
    pub hash: [u8; 32],

    /// Hash of the previous state root (None for genesis)
    pub parent: Option<[u8; 32]>,

    /// HLC timestamp when this root was created
    pub hlc: Hlc,

    /// Device ID that produced this state
    pub device_id: Uuid,

    /// Ed25519 signature over (hash || parent || hlc || device_id || seq)
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
        hlc: Hlc,
        device_id: Uuid,
        seq: u64,
    ) -> Self {
        Self {
            hash,
            parent,
            hlc,
            device_id,
            signature: [0u8; 64], // Unsigned
            seq,
        }
    }

    /// Get the bytes to be signed.
    ///
    /// Layout: hash (32) || parent (32, zeroed if None) || hlc (26) || device_id (16) || seq (8)
    /// Total: 114 bytes
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(114);
        bytes.extend_from_slice(&self.hash);
        if let Some(parent) = &self.parent {
            bytes.extend_from_slice(parent);
        } else {
            bytes.extend_from_slice(&[0u8; 32]);
        }
        bytes.extend_from_slice(&self.hlc.to_bytes());
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

    fn test_hlc() -> Hlc {
        Hlc::new(1234567890, [0u8; 16])
    }

    #[test]
    fn test_state_root_creation() {
        let root = StateRoot::new(
            [1u8; 32],
            None,
            test_hlc(),
            Uuid::from_bytes([1u8; 16]),
            0,
        );

        assert!(root.is_genesis());
        assert_eq!(root.seq, 0);
    }

    #[test]
    fn test_state_chain() {
        let genesis = StateRoot::new(
            [1u8; 32],
            None,
            Hlc::new(1000, [0u8; 16]),
            Uuid::from_bytes([1u8; 16]),
            0,
        );

        let second = StateRoot::new(
            [2u8; 32],
            Some(genesis.hash),
            Hlc::new(2000, [0u8; 16]),
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
            test_hlc(),
            Uuid::from_bytes([0u8; 16]),
            5,
        );

        let bytes1 = root.signing_bytes();
        let bytes2 = root.signing_bytes();

        assert_eq!(bytes1, bytes2);
        // hash(32) + parent(32) + hlc(26) + device_id(16) + seq(8) = 114
        assert_eq!(bytes1.len(), 114);
    }

    #[test]
    fn test_signing_bytes_includes_hlc() {
        let root1 = StateRoot::new(
            [1u8; 32],
            None,
            Hlc::new(1000, [0u8; 16]),
            Uuid::from_bytes([0u8; 16]),
            0,
        );

        let root2 = StateRoot::new(
            [1u8; 32],
            None,
            Hlc::new(2000, [0u8; 16]),
            Uuid::from_bytes([0u8; 16]),
            0,
        );

        // Different HLC → different signing bytes
        assert_ne!(root1.signing_bytes(), root2.signing_bytes());
    }
}
