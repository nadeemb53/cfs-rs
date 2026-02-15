//! CP Sync - Merkle trees, cognitive diffs, and crypto
//!
//! Provides:
//! - Diff generation and application
//! - Encryption (XChaCha20-Poly1305)
//! - Signatures (Ed25519)
//! - Device identity and pairing (X25519 key agreement)
//! - Serialization (CBOR + zstd compression)

mod crypto;
mod merkle;
mod identity;

pub use crypto::CryptoEngine;
pub use merkle::MerkleTree;
pub use identity::{DeviceIdentity, PairedDevice, PairingRequest, PairingConfirmation};

use cp_core::{CognitiveDiff, CPError, Result};

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

/// Signed diff wrapper for relay transmission
///
/// Per CP-013: Wraps an encrypted diff with authentication data
/// for secure relay-based synchronization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedDiff {
    /// The encrypted diff payload
    pub encrypted_diff: EncryptedPayload,

    /// Ed25519 signature over: nonce || ciphertext || sender_id || target_id || sequence
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],

    /// Sender's Ed25519 public key
    pub sender_public_key: [u8; 32],

    /// Sender's device ID (BLAKE3-16 of public key)
    pub sender_device_id: [u8; 16],

    /// Target device ID (for routing)
    pub target_device_id: [u8; 16],

    /// Sequence number for ordering
    pub sequence: u64,
}

impl SignedDiff {
    /// Create a signed diff from an encrypted payload
    pub fn new(
        identity: &DeviceIdentity,
        encrypted_diff: EncryptedPayload,
        target_device_id: [u8; 16],
        sequence: u64,
    ) -> Self {
        let signing_data = Self::compute_signing_data(
            &encrypted_diff.nonce,
            &encrypted_diff.ciphertext,
            &identity.device_id,
            &target_device_id,
            sequence,
        );

        let signature = identity.sign(&signing_data);

        Self {
            encrypted_diff,
            signature,
            sender_public_key: identity.public_key,
            sender_device_id: identity.device_id,
            target_device_id,
            sequence,
        }
    }

    /// Compute the data to be signed
    fn compute_signing_data(
        nonce: &[u8; 24],
        ciphertext: &[u8],
        sender_id: &[u8; 16],
        target_id: &[u8; 16],
        sequence: u64,
    ) -> Vec<u8> {
        let mut data = Vec::with_capacity(24 + ciphertext.len() + 16 + 16 + 8);
        data.extend_from_slice(nonce);
        data.extend_from_slice(ciphertext);
        data.extend_from_slice(sender_id);
        data.extend_from_slice(target_id);
        data.extend_from_slice(&sequence.to_le_bytes());
        data
    }

    /// Verify the signature on this signed diff
    pub fn verify(&self) -> Result<()> {
        use ed25519_dalek::{Signature, VerifyingKey, Verifier};

        let verifying_key = VerifyingKey::from_bytes(&self.sender_public_key)
            .map_err(|e| CPError::Crypto(format!("Invalid public key: {}", e)))?;

        // Verify that sender_device_id matches public key
        let expected_device_id: [u8; 16] = blake3::hash(&self.sender_public_key).as_bytes()[0..16]
            .try_into()
            .unwrap();
        if expected_device_id != self.sender_device_id {
            return Err(CPError::Verification("Device ID doesn't match public key".into()));
        }

        let signing_data = Self::compute_signing_data(
            &self.encrypted_diff.nonce,
            &self.encrypted_diff.ciphertext,
            &self.sender_device_id,
            &self.target_device_id,
            self.sequence,
        );

        let signature = Signature::from_bytes(&self.signature);

        verifying_key
            .verify(&signing_data, &signature)
            .map_err(|_| CPError::Verification("Invalid SignedDiff signature".into()))
    }

    /// Get sender device ID as hex string
    pub fn sender_device_id_hex(&self) -> String {
        self.sender_device_id
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }

    /// Get target device ID as hex string
    pub fn target_device_id_hex(&self) -> String {
        self.target_device_id
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

/// Serialize a diff to CBOR and compress
pub fn serialize_diff(diff: &CognitiveDiff) -> Result<Vec<u8>> {
    let mut cbor_bytes = Vec::new();
    ciborium::into_writer(diff, &mut cbor_bytes)
        .map_err(|e| cp_core::CPError::Serialization(e.to_string()))?;
    
    let compressed = zstd::encode_all(cbor_bytes.as_slice(), 3)
        .map_err(|e| cp_core::CPError::Serialization(e.to_string()))?;
    
    Ok(compressed)
}

/// Decompress and deserialize a diff from CBOR
pub fn deserialize_diff(data: &[u8]) -> Result<CognitiveDiff> {
    let decompressed = zstd::decode_all(data)
        .map_err(|e| cp_core::CPError::Serialization(e.to_string()))?;

    let diff: CognitiveDiff = ciborium::from_reader(decompressed.as_slice())
        .map_err(|e| cp_core::CPError::Serialization(e.to_string()))?;

    Ok(diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::Hlc;
    use uuid::Uuid;

    #[test]
    fn test_signed_diff_roundtrip() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Create a simple diff
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::from_bytes(alice.device_id),
            1,
            Hlc::new(1000, alice.device_id),
        );

        // Alice encrypts and signs for Bob
        let crypto = CryptoEngine::new();
        let encrypted = crypto.encrypt_diff(&diff).unwrap();

        let signed = SignedDiff::new(&alice, encrypted, bob.device_id, 1);

        // Verify signature
        assert!(signed.verify().is_ok());

        // Check device IDs
        assert_eq!(signed.sender_device_id, alice.device_id);
        assert_eq!(signed.target_device_id, bob.device_id);
        assert_eq!(signed.sequence, 1);
    }

    #[test]
    fn test_signed_diff_verification_fails_on_tamper() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::from_bytes(alice.device_id),
            1,
            Hlc::new(1000, alice.device_id),
        );

        let crypto = CryptoEngine::new();
        let encrypted = crypto.encrypt_diff(&diff).unwrap();

        let mut signed = SignedDiff::new(&alice, encrypted, bob.device_id, 1);

        // Tamper with the sequence
        signed.sequence = 2;

        // Verification should fail
        assert!(signed.verify().is_err());
    }
}
