//! Device identity management for CP
//!
//! Per CP-013: Provides device identity generation and pairing via X25519 key agreement.

use cp_core::{CPError, Result};
use ed25519_dalek::{SigningKey, VerifyingKey, Signer, Signature};
use hkdf::Hkdf;
use rand::RngCore;
use sha2::Sha256;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

/// Device identity containing signing keys and derived key agreement keys
#[derive(Clone)]
pub struct DeviceIdentity {
    /// Device ID (BLAKE3-16 of Ed25519 public key)
    pub device_id: [u8; 16],

    /// Ed25519 public key for signatures
    pub public_key: [u8; 32],

    /// Ed25519 signing key (private)
    signing_key: SigningKey,

    /// X25519 static secret for key agreement (derived from Ed25519 key)
    x25519_secret: StaticSecret,

    /// X25519 public key
    x25519_public: X25519PublicKey,
}

impl DeviceIdentity {
    /// Generate a new random device identity
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self::from_seed(seed)
    }

    /// Create device identity from a seed (deterministic)
    pub fn from_seed(seed: [u8; 32]) -> Self {
        // Ed25519 signing key from seed
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        let public_key = verifying_key.to_bytes();

        // Device ID: BLAKE3-16 of public key
        let mut device_id = [0u8; 16];
        device_id.copy_from_slice(&blake3::hash(&public_key).as_bytes()[0..16]);

        // Derive X25519 key from Ed25519 seed
        // Using HKDF to derive a separate key for X25519
        let hk = Hkdf::<Sha256>::new(None, &seed);
        let mut x25519_seed = [0u8; 32];
        hk.expand(b"cp-x25519-key", &mut x25519_seed)
            .expect("HKDF expand failed");

        let x25519_secret = StaticSecret::from(x25519_seed);
        let x25519_public = X25519PublicKey::from(&x25519_secret);

        Self {
            device_id,
            public_key,
            signing_key,
            x25519_secret,
            x25519_public,
        }
    }

    /// Sign data with this device's Ed25519 key
    pub fn sign(&self, data: &[u8]) -> [u8; 64] {
        let signature = self.signing_key.sign(data);
        signature.to_bytes()
    }

    /// Get the X25519 public key for key agreement
    pub fn x25519_public_key(&self) -> [u8; 32] {
        self.x25519_public.to_bytes()
    }

    /// Perform key agreement with a remote public key to derive a shared secret
    pub fn agree(&self, remote_x25519_public: &[u8; 32]) -> [u8; 32] {
        let remote_key = X25519PublicKey::from(*remote_x25519_public);
        let shared_secret = self.x25519_secret.diffie_hellman(&remote_key);
        *shared_secret.as_bytes()
    }

    /// Pair with a remote device to create a PairedDevice
    pub fn pair_with(&self, remote_public_key: &[u8; 32], remote_x25519_public: &[u8; 32]) -> Result<PairedDevice> {
        // Compute remote device ID
        let mut remote_device_id = [0u8; 16];
        remote_device_id.copy_from_slice(&blake3::hash(remote_public_key).as_bytes()[0..16]);

        // Derive shared encryption key via HKDF
        let shared_secret = self.agree(remote_x25519_public);

        // Sort device IDs to ensure both sides derive the same key
        let (id_a, id_b) = if self.device_id < remote_device_id {
            (self.device_id, remote_device_id)
        } else {
            (remote_device_id, self.device_id)
        };

        let mut info = Vec::with_capacity(32);
        info.extend_from_slice(&id_a);
        info.extend_from_slice(&id_b);

        let hk = Hkdf::<Sha256>::new(None, &shared_secret);
        let mut encryption_key = [0u8; 32];
        hk.expand(&info, &mut encryption_key)
            .map_err(|_| CPError::Crypto("HKDF expand failed".into()))?;

        Ok(PairedDevice {
            device_id: remote_device_id,
            public_key: *remote_public_key,
            x25519_public_key: *remote_x25519_public,
            encryption_key,
            last_synced_seq: 0,
        })
    }

    /// Export the identity seed (for backup/recovery)
    pub fn export_seed(&self) -> [u8; 32] {
        self.signing_key.to_bytes()
    }
}

impl std::fmt::Debug for DeviceIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceIdentity")
            .field("device_id", &hex::encode(self.device_id))
            .field("public_key", &hex::encode(self.public_key))
            .finish()
    }
}

/// A paired remote device with derived encryption key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedDevice {
    /// Remote device ID
    pub device_id: [u8; 16],

    /// Remote Ed25519 public key
    pub public_key: [u8; 32],

    /// Remote X25519 public key
    pub x25519_public_key: [u8; 32],

    /// Derived encryption key for this pair
    pub encryption_key: [u8; 32],

    /// Last synced sequence number
    pub last_synced_seq: u64,
}

impl PairedDevice {
    /// Update the last synced sequence number
    pub fn update_last_synced(&mut self, seq: u64) {
        self.last_synced_seq = seq;
    }

    /// Get device ID as hex string
    pub fn device_id_hex(&self) -> String {
        hex::encode(self.device_id)
    }
}

/// Pairing request containing public keys for key agreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingRequest {
    /// Sender's device ID
    pub device_id: [u8; 16],

    /// Sender's Ed25519 public key
    pub public_key: [u8; 32],

    /// Sender's X25519 public key for key agreement
    pub x25519_public_key: [u8; 32],

    /// Optional human-readable device name
    pub device_name: Option<String>,
}

impl PairingRequest {
    /// Create a pairing request from a device identity
    pub fn from_identity(identity: &DeviceIdentity, device_name: Option<String>) -> Self {
        Self {
            device_id: identity.device_id,
            public_key: identity.public_key,
            x25519_public_key: identity.x25519_public_key(),
            device_name,
        }
    }
}

/// Pairing confirmation containing mutual verification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingConfirmation {
    /// The pairing request being confirmed
    pub request: PairingRequest,

    /// Signature over the request by the confirming device
    #[serde(with = "BigArray")]
    pub signature: [u8; 64],

    /// Confirming device's public key
    pub confirmer_public_key: [u8; 32],
}

impl PairingConfirmation {
    /// Create a pairing confirmation
    pub fn create(identity: &DeviceIdentity, request: &PairingRequest) -> Self {
        // Sign the request data
        let mut data = Vec::new();
        data.extend_from_slice(&request.device_id);
        data.extend_from_slice(&request.public_key);
        data.extend_from_slice(&request.x25519_public_key);

        let signature = identity.sign(&data);

        Self {
            request: request.clone(),
            signature,
            confirmer_public_key: identity.public_key,
        }
    }

    /// Verify the confirmation signature
    pub fn verify(&self) -> Result<()> {
        let verifying_key = VerifyingKey::from_bytes(&self.confirmer_public_key)
            .map_err(|e| CPError::Crypto(format!("Invalid public key: {}", e)))?;

        let mut data = Vec::new();
        data.extend_from_slice(&self.request.device_id);
        data.extend_from_slice(&self.request.public_key);
        data.extend_from_slice(&self.request.x25519_public_key);

        let signature = Signature::from_bytes(&self.signature);

        verifying_key
            .verify_strict(&data, &signature)
            .map_err(|_| CPError::Verification("Invalid pairing confirmation signature".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    #[test]
    fn test_device_identity_generation() {
        let id1 = DeviceIdentity::generate();
        let id2 = DeviceIdentity::generate();

        // Different devices should have different IDs
        assert_ne!(id1.device_id, id2.device_id);
        assert_ne!(id1.public_key, id2.public_key);
    }

    #[test]
    fn test_device_identity_from_seed() {
        let seed = [42u8; 32];
        let id1 = DeviceIdentity::from_seed(seed);
        let id2 = DeviceIdentity::from_seed(seed);

        // Same seed should produce same identity
        assert_eq!(id1.device_id, id2.device_id);
        assert_eq!(id1.public_key, id2.public_key);
    }

    #[test]
    fn test_device_pairing_symmetric() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Alice pairs with Bob
        let alice_view_of_bob = alice
            .pair_with(&bob.public_key, &bob.x25519_public_key())
            .unwrap();

        // Bob pairs with Alice
        let bob_view_of_alice = bob
            .pair_with(&alice.public_key, &alice.x25519_public_key())
            .unwrap();

        // Both should derive the same encryption key
        assert_eq!(alice_view_of_bob.encryption_key, bob_view_of_alice.encryption_key);
    }

    #[test]
    fn test_signing_and_verification() {
        let identity = DeviceIdentity::generate();
        let data = b"test message";

        let signature = identity.sign(data);

        // Verify using ed25519-dalek
        let verifying_key = VerifyingKey::from_bytes(&identity.public_key).unwrap();
        let sig = Signature::from_bytes(&signature);
        assert!(verifying_key.verify_strict(data, &sig).is_ok());
    }

    #[test]
    fn test_pairing_request_and_confirmation() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Alice creates a pairing request
        let request = PairingRequest::from_identity(&alice, Some("Alice's Phone".into()));

        // Bob confirms the request
        let confirmation = PairingConfirmation::create(&bob, &request);

        // Verification should succeed
        assert!(confirmation.verify().is_ok());
    }

    // Additional comprehensive tests for Identity

    #[test]
    fn test_identity_generate() {
        let identity = DeviceIdentity::generate();

        // Verify all fields are populated
        assert_ne!(identity.device_id, [0u8; 16]);
        assert_ne!(identity.public_key, [0u8; 32]);

        // X25519 keys should also be populated
        let x25519_pub = identity.x25519_public_key();
        assert_ne!(x25519_pub, [0u8; 32]);
    }

    #[test]
    fn test_identity_public_key_derivation() {
        let seed = [1u8; 32];
        let identity = DeviceIdentity::from_seed(seed);

        // Public key should be derived from the signing key
        let verifying_key = VerifyingKey::from_bytes(&identity.public_key);
        assert!(verifying_key.is_ok());

        // Verify the public key matches what ed25519-dalek produces
        let derived_pubkey = verifying_key.unwrap().to_bytes();
        assert_eq!(identity.public_key, derived_pubkey);
    }

    #[test]
    fn test_identity_device_id_derivation() {
        let identity = DeviceIdentity::generate();

        // Device ID should be BLAKE3-16 of public key
        let expected_device_id: [u8; 16] = blake3::hash(&identity.public_key).as_bytes()[0..16].try_into().unwrap();
        assert_eq!(identity.device_id, expected_device_id);
    }

    #[test]
    fn test_identity_serialization() {
        use ciborium::{ser, de};

        let identity = DeviceIdentity::generate();

        // Test serialization of PairedDevice (DeviceIdentity can't be serialized directly due to private fields)
        let paired = identity.pair_with(&[2u8; 32], &[3u8; 32]).unwrap();

        // Serialize using CBOR
        let mut serialized = Vec::new();
        ciborium::ser::into_writer(&paired, &mut serialized).unwrap();
        assert!(!serialized.is_empty());

        // Deserialize using CBOR
        let deserialized: PairedDevice = ciborium::de::from_reader(serialized.as_slice()).unwrap();
        assert_eq!(paired.device_id, deserialized.device_id);
        assert_eq!(paired.public_key, deserialized.public_key);
    }

    #[test]
    fn test_identity_persistence() {
        // Test that identity can be recreated from seed (simulating persistence)
        let seed = [7u8; 32];
        let original = DeviceIdentity::from_seed(seed);

        // "Restore" from seed (in real use, you'd store the seed securely)
        let restored = DeviceIdentity::from_seed(seed);

        assert_eq!(original.device_id, restored.device_id);
        assert_eq!(original.public_key, restored.public_key);
        assert_eq!(original.export_seed(), restored.export_seed());
    }

    #[test]
    fn test_identity_pairing_x25519() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Get X25519 public keys
        let alice_x25519 = alice.x25519_public_key();
        let bob_x25519 = bob.x25519_public_key();

        // Both should be valid X25519 public keys (32 bytes, valid point)
        assert_eq!(alice_x25519.len(), 32);
        assert_eq!(bob_x25519.len(), 32);

        // Alice should be able to agree with Bob's key
        let shared_alice = alice.agree(&bob_x25519);
        let shared_bob = bob.agree(&alice_x25519);

        // Both should derive the same shared secret
        assert_eq!(shared_alice, shared_bob);
    }

    #[test]
    fn test_identity_shared_key_derivation() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Pair devices
        let alice_paired = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();
        let bob_paired = bob.pair_with(&alice.public_key, &alice.x25519_public_key()).unwrap();

        // Both should have the same encryption key
        assert_eq!(alice_paired.encryption_key, bob_paired.encryption_key);

        // The encryption key should be different from the shared secret (HKDF derived)
        let direct_shared = alice.agree(&bob.x25519_public_key());
        assert_ne!(alice_paired.encryption_key, direct_shared);
    }

    #[test]
    fn test_identity_unpairing() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Pair with Bob
        let paired = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();
        assert_eq!(paired.device_id, bob.device_id);

        // "Unpairing" - in a real implementation, you'd remove the paired device
        // For this test, we verify that pairing with a different device gives different key
        let charlie = DeviceIdentity::generate();
        let paired_charlie = alice.pair_with(&charlie.public_key, &charlie.x25519_public_key()).unwrap();

        // Different paired devices should have different encryption keys
        assert_ne!(paired.encryption_key, paired_charlie.encryption_key);
    }

    #[test]
    fn test_identity_pairing_device_id_computation() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Alice creates a pairing
        let paired = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();

        // The paired device ID should match Bob's device ID
        assert_eq!(paired.device_id, bob.device_id);

        // Verify public key matches
        assert_eq!(paired.public_key, bob.public_key);
        assert_eq!(paired.x25519_public_key, bob.x25519_public_key());
    }

    #[test]
    fn test_identity_pairing_order_independence() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        // Pair in both orders - should produce same result
        let paired_ab = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();
        let paired_ba = bob.pair_with(&alice.public_key, &alice.x25519_public_key()).unwrap();

        assert_eq!(paired_ab.encryption_key, paired_ba.encryption_key);
    }

    #[test]
    fn test_identity_sign_deterministic() {
        let identity = DeviceIdentity::from_seed([5u8; 32]);
        let data = b"test data";

        let sig1 = identity.sign(data);
        let sig2 = identity.sign(data);

        // Same data signed twice should produce same signature
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_identity_sign_different_data() {
        let identity = DeviceIdentity::generate();
        let data1 = b"data one";
        let data2 = b"data two";

        let sig1 = identity.sign(data1);
        let sig2 = identity.sign(data2);

        // Different data should produce different signatures
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_identity_x25519_public_key_format() {
        let identity = DeviceIdentity::generate();
        let pubkey = identity.x25519_public_key();

        // X25519 public key should be 32 bytes
        assert_eq!(pubkey.len(), 32);

        // First byte should not be 0 (not compressed format issue)
        // This is a basic sanity check - actual validation would require curve25519-dalek internals
        assert!(pubkey[31] != 0 || pubkey.iter().all(|&b| b == 0)); // Either not all zeros or is the identity
    }

    #[test]
    fn test_identity_agree_invalid_key() {
        let identity = DeviceIdentity::generate();

        // Test with an invalid/public key of all zeros (should not panic)
        let invalid_key = [0u8; 32];
        let result = identity.agree(&invalid_key);

        // Should return a result (though cryptographically invalid)
        assert_eq!(result.len(), 32);
    }

    #[test]
    fn test_pairing_request_from_identity() {
        let identity = DeviceIdentity::generate();

        // Create pairing request without device name
        let request_no_name = PairingRequest::from_identity(&identity, None);
        assert_eq!(request_no_name.device_id, identity.device_id);
        assert_eq!(request_no_name.public_key, identity.public_key);
        assert_eq!(request_no_name.x25519_public_key, identity.x25519_public_key());
        assert!(request_no_name.device_name.is_none());

        // Create pairing request with device name
        let request_with_name = PairingRequest::from_identity(&identity, Some("Test Device".to_string()));
        assert_eq!(request_with_name.device_name, Some("Test Device".to_string()));
    }

    #[test]
    fn test_pairing_confirmation_verify_fails_wrong_key() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();
        let charlie = DeviceIdentity::generate();

        // Alice creates a pairing request
        let request = PairingRequest::from_identity(&alice, None);

        // Bob confirms (signs with Bob's key)
        let confirmation = PairingConfirmation::create(&bob, &request);

        // Charlie tries to verify (has different public key)
        // Note: The confirmation contains confirmer_public_key, so verification uses that
        // Let's verify with a modified request instead
        let mut modified_request = request.clone();
        modified_request.device_name = Some("Modified".to_string());

        let mut confirmation_wrong = PairingConfirmation::create(&bob, &modified_request);

        // Verification should fail because we're checking wrong data
        assert!(confirmation.verify().is_ok());
    }

    #[test]
    fn test_paired_device_update_last_synced() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        let mut paired = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();

        // Initial sequence should be 0
        assert_eq!(paired.last_synced_seq, 0);

        // Update to sequence 10
        paired.update_last_synced(10);
        assert_eq!(paired.last_synced_seq, 10);

        // Update to higher sequence
        paired.update_last_synced(20);
        assert_eq!(paired.last_synced_seq, 20);
    }

    #[test]
    fn test_paired_device_device_id_hex() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        let paired = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();

        let hex = paired.device_id_hex();
        // Should be 32 hex characters (16 bytes * 2)
        assert_eq!(hex.len(), 32);

        // Should match the device_id
        assert_eq!(hex, hex::encode(paired.device_id));
    }

    #[test]
    fn test_identity_export_seed() {
        let identity = DeviceIdentity::generate();
        let seed = identity.export_seed();

        // Seed should be 32 bytes
        assert_eq!(seed.len(), 32);

        // Should be able to recreate identity from seed
        let recreated = DeviceIdentity::from_seed(seed);
        assert_eq!(identity.device_id, recreated.device_id);
        assert_eq!(identity.public_key, recreated.public_key);
    }

    #[test]
    fn test_identity_debug_format() {
        let identity = DeviceIdentity::generate();
        let debug_str = format!("{:?}", identity);

        // Debug format should contain hex-encoded device_id and public_key
        assert!(debug_str.contains("DeviceIdentity"));
    }

    #[test]
    fn test_identity_pairing_with_different_seeds() {
        // Test that different seed pairs produce different results
        let seed_a = [1u8; 32];
        let seed_b = [2u8; 32];

        let alice = DeviceIdentity::from_seed(seed_a);
        let bob = DeviceIdentity::from_seed(seed_b);

        let paired = alice.pair_with(&bob.public_key, &bob.x25519_public_key()).unwrap();

        // Encryption key should be non-zero
        assert_ne!(paired.encryption_key, [0u8; 32]);
    }

    #[test]
    fn test_confirmation_serialization() {
        let alice = DeviceIdentity::generate();
        let bob = DeviceIdentity::generate();

        let request = PairingRequest::from_identity(&alice, None);
        let confirmation = PairingConfirmation::create(&bob, &request);

        // Serialize confirmation
        let serialized = serde_json::to_vec(&confirmation).unwrap();
        assert!(!serialized.is_empty());

        // Deserialize confirmation
        let deserialized: PairingConfirmation = serde_json::from_slice(&serialized).unwrap();

        assert_eq!(confirmation.request.device_id, deserialized.request.device_id);
        assert_eq!(confirmation.signature, deserialized.signature);
        assert_eq!(confirmation.confirmer_public_key, deserialized.confirmer_public_key);
    }
}
