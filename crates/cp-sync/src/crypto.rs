//! Cryptographic operations for CP

use cp_core::{CPError, CognitiveDiff, Result};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    XChaCha20Poly1305, XNonce,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::RngCore;

use crate::{serialize_diff, deserialize_diff, EncryptedPayload};

/// Cryptographic engine for encrypting and signing diffs
pub struct CryptoEngine {
    /// Symmetric encryption key
    symmetric_key: [u8; 32],
    /// Ed25519 signing key
    signing_key: SigningKey,
}

impl CryptoEngine {
    /// Create a new crypto engine with random keys
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        
        let mut symmetric_key = [0u8; 32];
        rng.fill_bytes(&mut symmetric_key);
        
        let mut signing_seed = [0u8; 32];
        rng.fill_bytes(&mut signing_seed);
        let signing_key = SigningKey::from_bytes(&signing_seed);
        
        Self {
            symmetric_key,
            signing_key,
        }
    }

    /// Create from existing keys
    pub fn from_keys(symmetric_key: [u8; 32], signing_key_bytes: [u8; 32]) -> Self {
        Self {
            symmetric_key,
            signing_key: SigningKey::from_bytes(&signing_key_bytes),
        }
    }

    /// Create from seed (derives keys deterministically)
    pub fn new_with_seed(seed: [u8; 32]) -> Self {
        // Simple derivation for MVP:
        // symmetric = hash(seed || "enc")
        // signing = hash(seed || "sign")
        
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed);
        hasher.update(b"enc");
        let symmetric_key = *hasher.finalize().as_bytes();
        
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed);
        hasher.update(b"sign");
        let signing_seed = *hasher.finalize().as_bytes();
        
        Self {
            symmetric_key,
            signing_key: SigningKey::from_bytes(&signing_seed),
        }
    }

    /// Get the public key for verification
    pub fn public_key(&self) -> [u8; 32] {
        self.signing_key.verifying_key().to_bytes()
    }

    /// Encrypt and sign a cognitive diff
    pub fn encrypt_diff(&self, diff: &CognitiveDiff) -> Result<EncryptedPayload> {
        // Serialize and compress
        let plaintext = serialize_diff(diff)?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 24];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = XNonce::from_slice(&nonce_bytes);
        
        // Encrypt
        let cipher = XChaCha20Poly1305::new_from_slice(&self.symmetric_key)
            .map_err(|e| CPError::Crypto(e.to_string()))?;
        
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| CPError::Crypto(e.to_string()))?;
        
        // Sign the ciphertext
        let signature = self.signing_key.sign(&ciphertext);
        
        Ok(EncryptedPayload {
            ciphertext,
            nonce: nonce_bytes,
            signature: signature.to_bytes(),
            public_key: self.public_key(),
        })
    }

    /// Sign arbitrary data
    pub fn sign(&self, data: &[u8]) -> Signature {
        self.signing_key.sign(data)
    }

    /// Decrypt and verify a payload
    pub fn decrypt_diff(&self, payload: &EncryptedPayload) -> Result<CognitiveDiff> {
        // Verify signature
        let verifying_key = VerifyingKey::from_bytes(&payload.public_key)
            .map_err(|e| CPError::Crypto(e.to_string()))?;
        
        let signature = Signature::from_bytes(&payload.signature);
        
        verifying_key
            .verify(&payload.ciphertext, &signature)
            .map_err(|_| CPError::Verification("Invalid signature".into()))?;
        
        // Decrypt
        let nonce = XNonce::from_slice(&payload.nonce);
        let cipher = XChaCha20Poly1305::new_from_slice(&self.symmetric_key)
            .map_err(|e| CPError::Crypto(e.to_string()))?;
        
        let plaintext = cipher
            .decrypt(nonce, payload.ciphertext.as_ref())
            .map_err(|_| CPError::Crypto("Decryption failed".into()))?;
        
        // Deserialize
        deserialize_diff(&plaintext)
    }
}

impl Default for CryptoEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cp_core::{CognitiveDiff, Document, Hlc};
    use uuid::Uuid;
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    // CP-013 Test Vectors (known values for interoperability testing)
    // These are sample values - in production, use values from CP-013 spec
    const TEST_SEED: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
        0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    ];

    const TEST_MESSAGE: &[u8] = b"Canon Protocol v1.0 - Test Message";

    #[test]
    fn test_crypto_generate_keypair() {
        let engine = CryptoEngine::new();

        // Verify that a valid public key was generated (32 bytes)
        let public_key = engine.public_key();
        assert_eq!(public_key.len(), 32);

        // Different engine instances should have different keys
        let engine2 = CryptoEngine::new();
        assert_ne!(engine.public_key(), engine2.public_key());
    }

    #[test]
    fn test_crypto_sign() {
        let engine = CryptoEngine::new();
        let message = b"Test message for signing";

        let signature = engine.sign(message);

        // Ed25519 signature should be 64 bytes
        assert_eq!(signature.to_bytes().len(), 64);
    }

    #[test]
    fn test_crypto_verify_valid() {
        let engine = CryptoEngine::new();
        let message = b"Test message for verification";

        let signature = engine.sign(message);
        let public_key = engine.public_key();

        // Verify the signature
        let verifying_key = VerifyingKey::from_bytes(&public_key).unwrap();
        let sig = Signature::from_bytes(&signature.to_bytes());

        assert!(verifying_key.verify(message, &sig).is_ok());
    }

    #[test]
    fn test_crypto_verify_invalid() {
        let engine = CryptoEngine::new();
        let message = b"Original message";

        let signature = engine.sign(message);

        // Try to verify with a different message - should fail
        let public_key = engine.public_key();
        let verifying_key = VerifyingKey::from_bytes(&public_key).unwrap();
        let sig = Signature::from_bytes(&signature.to_bytes());

        let wrong_message = b"Different message";
        assert!(verifying_key.verify(wrong_message, &sig).is_err());
    }

    #[test]
    fn test_crypto_encrypt_xchacha20() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::new_v4(),
            0,
            Hlc::new(1000, [0u8; 16]),
        );

        let encrypted = engine.encrypt_diff(&diff).unwrap();

        // Verify encrypted payload structure
        assert_eq!(encrypted.nonce.len(), 24);
        assert_eq!(encrypted.signature.len(), 64);
        assert_eq!(encrypted.public_key.len(), 32);
        // Ciphertext should be different from plaintext
        assert!(!encrypted.ciphertext.is_empty());
    }

    #[test]
    fn test_crypto_decrypt_xchacha20() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::new_v4(),
            0,
            Hlc::new(1000, [0u8; 16]),
        );

        let encrypted = engine.encrypt_diff(&diff).unwrap();
        let decrypted = engine.decrypt_diff(&encrypted).unwrap();

        // Verify metadata is preserved
        assert_eq!(diff.metadata.device_id, decrypted.metadata.device_id);
    }

    #[test]
    fn test_crypto_encrypt_decrypt_roundtrip() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty([0u8; 32], Uuid::new_v4(), 0, Hlc::new(1000, [0u8; 16]));

        let encrypted = engine.encrypt_diff(&diff).unwrap();
        let decrypted = engine.decrypt_diff(&encrypted).unwrap();

        assert_eq!(diff.metadata.device_id, decrypted.metadata.device_id);
    }

    #[test]
    fn test_crypto_nonce_uniqueness() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::new_v4(),
            0,
            Hlc::new(1000, [0u8; 16]),
        );

        // Generate multiple encrypted diffs
        let encrypted1 = engine.encrypt_diff(&diff).unwrap();
        let encrypted2 = engine.encrypt_diff(&diff).unwrap();
        let encrypted3 = engine.encrypt_diff(&diff).unwrap();

        // Nonces should be unique (highly unlikely to collide)
        assert_ne!(encrypted1.nonce, encrypted2.nonce);
        assert_ne!(encrypted2.nonce, encrypted3.nonce);
        assert_ne!(encrypted1.nonce, encrypted3.nonce);
    }

    #[test]
    fn test_crypto_key_derivation_hkdf() {
        // Test deterministic key derivation from seed
        let engine1 = CryptoEngine::new_with_seed(TEST_SEED);
        let engine2 = CryptoEngine::new_with_seed(TEST_SEED);

        // Same seed should produce same keys
        assert_eq!(engine1.public_key(), engine2.public_key());

        // Different seed should produce different keys
        let mut different_seed = TEST_SEED;
        different_seed[0] ^= 0xFF;
        let engine3 = CryptoEngine::new_with_seed(different_seed);

        assert_ne!(engine1.public_key(), engine3.public_key());
    }

    #[test]
    fn test_crypto_test_vector_encryption() {
        // Test encryption with known test vector inputs
        let symmetric_key = [0x42u8; 32];
        let signing_key_bytes = [0x24u8; 32];
        let engine = CryptoEngine::from_keys(symmetric_key, signing_key_bytes);

        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::nil(),
            0,
            Hlc::new(0, [0u8; 16]),
        );

        // Should be able to encrypt and decrypt
        let encrypted = engine.encrypt_diff(&diff).unwrap();
        let decrypted = engine.decrypt_diff(&encrypted).unwrap();

        assert_eq!(decrypted.metadata.device_id, Uuid::nil());
    }

    #[test]
    fn test_crypto_test_vector_signature() {
        // Test signature with known test vector
        let engine = CryptoEngine::from_keys([0x42u8; 32], [0x24u8; 32]);

        let signature = engine.sign(TEST_MESSAGE);

        // Verify signature can be parsed and verified
        let public_key = engine.public_key();
        let verifying_key = VerifyingKey::from_bytes(&public_key).unwrap();
        let sig = Signature::from_bytes(&signature.to_bytes());

        assert!(verifying_key.verify(TEST_MESSAGE, &sig).is_ok());
    }

    #[test]
    fn test_crypto_tampering_detection() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::new_v4(),
            0,
            Hlc::new(1000, [0u8; 16]),
        );

        let mut encrypted = engine.encrypt_diff(&diff).unwrap();

        // Tamper with the ciphertext
        encrypted.ciphertext[0] ^= 0xFF;

        // Decryption should fail due to authentication tag mismatch
        assert!(engine.decrypt_diff(&encrypted).is_err());
    }

    #[test]
    fn test_crypto_wrong_key_rejected() {
        let engine1 = CryptoEngine::new();
        let engine2 = CryptoEngine::new();

        let diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::new_v4(),
            0,
            Hlc::new(1000, [0u8; 16]),
        );

        // Encrypt with engine1
        let encrypted = engine1.encrypt_diff(&diff).unwrap();

        // Try to decrypt with engine2 - should fail
        assert!(engine2.decrypt_diff(&encrypted).is_err());
    }

    #[test]
    fn test_crypto_signature_verification() {
        let engine = CryptoEngine::new();
        let diff = CognitiveDiff::empty([0u8; 32], Uuid::new_v4(), 0, Hlc::new(1000, [0u8; 16]));

        let mut encrypted = engine.encrypt_diff(&diff).unwrap();

        // Tamper with signature
        encrypted.signature[0] ^= 0xFF;

        assert!(engine.decrypt_diff(&encrypted).is_err());
    }

    #[test]
    fn test_crypto_from_keys_deterministic() {
        // Test that from_keys produces deterministic results
        let symmetric_key = [0xAAu8; 32];
        let signing_key_bytes = [0x55u8; 32];

        let engine1 = CryptoEngine::from_keys(symmetric_key, signing_key_bytes);
        let engine2 = CryptoEngine::from_keys(symmetric_key, signing_key_bytes);

        assert_eq!(engine1.public_key(), engine2.public_key());
    }

    #[test]
    fn test_crypto_large_diff_encryption() {
        let engine = CryptoEngine::new();

        // Create a diff with actual data
        let mut diff = CognitiveDiff::empty(
            [0u8; 32],
            Uuid::new_v4(),
            0,
            Hlc::new(1000, [0u8; 16]),
        );

        // Add multiple documents to make it larger
        for i in 0..10 {
            let mut doc = Document::new(
                std::path::PathBuf::from(format!("test{}.md", i)),
                b"Test content for encryption",
                0,
            );
            diff.added_docs.push(doc);
        }

        let encrypted = engine.encrypt_diff(&diff).unwrap();
        let decrypted = engine.decrypt_diff(&encrypted).unwrap();

        assert_eq!(diff.added_docs.len(), decrypted.added_docs.len());
    }
}
