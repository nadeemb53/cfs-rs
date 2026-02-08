//! CFS Sync - Merkle trees, cognitive diffs, and crypto
//!
//! Provides:
//! - Diff generation and application
//! - Encryption (XChaCha20-Poly1305)
//! - Signatures (Ed25519)
//! - Serialization (CBOR + zstd compression)

mod crypto;
mod merkle;

pub use crypto::CryptoEngine;
pub use merkle::MerkleTree;

use cfs_core::{CognitiveDiff, Result};

use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Encrypted payload for transmission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedPayload {
    /// Encrypted CBOR data
    pub ciphertext: Vec<u8>,
    /// Nonce used for encryption
    #[serde(with = "BigArray")]
    pub nonce: [u8; 24],
    /// Signature over the ciphertext
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],
    /// Signer's public key
    pub public_key: [u8; 32],
}

/// Serialize a diff to CBOR and compress
pub fn serialize_diff(diff: &CognitiveDiff) -> Result<Vec<u8>> {
    let mut cbor_bytes = Vec::new();
    ciborium::into_writer(diff, &mut cbor_bytes)
        .map_err(|e| cfs_core::CfsError::Serialization(e.to_string()))?;
    
    let compressed = zstd::encode_all(cbor_bytes.as_slice(), 3)
        .map_err(|e| cfs_core::CfsError::Serialization(e.to_string()))?;
    
    Ok(compressed)
}

/// Decompress and deserialize a diff from CBOR
pub fn deserialize_diff(data: &[u8]) -> Result<CognitiveDiff> {
    let decompressed = zstd::decode_all(data)
        .map_err(|e| cfs_core::CfsError::Serialization(e.to_string()))?;
    
    let diff: CognitiveDiff = ciborium::from_reader(decompressed.as_slice())
        .map_err(|e| cfs_core::CfsError::Serialization(e.to_string()))?;
    
    Ok(diff)
}
