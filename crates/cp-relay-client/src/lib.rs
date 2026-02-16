//! CP Relay Client - HTTP client for sync operations
//!
//! Provides a strongly-typed client for the minimal relay server.
//! Handles encryption, serialization, and hex/base64 encoding.
//!
//! Per CP-013: Supports device-addressed routing with proper headers.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use cp_core::{CPError, Result, StateRoot};
use cp_sync::{EncryptedPayload, SignedDiff};
use serde::{Deserialize, Serialize};

/// Client for the CP relay server
pub struct RelayClient {
    base_url: String,
    auth_token: String,
    /// Local device ID for X-Device-ID header
    device_id: [u8; 16],
    client: reqwest::Client,
}

/// Request body for uploading a diff (matches relay server)
#[derive(Serialize)]
struct UploadRequest {
    ciphertext: String, // Base64
    nonce: String,      // Hex
    signature: String,  // Hex
    public_key: String, // Hex
    root: String,       // Hex
}

/// Response from listing roots (matches relay server)
#[derive(Deserialize)]
struct RootInfo {
    root: String,
    // timestamp: i64, // Unused for now
    // size: usize,    // Unused for now
}

// ============================================================================
// Device Pairing Types
// ============================================================================

/// Pairing initiation request
#[derive(Serialize)]
pub struct PairInitRequest {
    device_id: String,
    public_key: String,
    display_name: Option<String>,
}

/// Pairing initiation response
#[derive(Deserialize)]
pub struct PairInitResponse {
    pairing_id: String,
    pairing_code: String,
    expires_at: i64,
}

/// Pairing response request
#[derive(Serialize)]
pub struct PairRespondRequest {
    pairing_id: String,
    device_id: String,
    public_key: String,
    display_name: Option<String>,
}

/// Pairing confirmation request
#[derive(Serialize)]
pub struct PairConfirmRequest {
    pairing_id: String,
}

/// Pairing confirmation response
#[derive(Deserialize)]
pub struct PairConfirmResponse {
    success: bool,
    peer_device_id: String,
    peer_display_name: Option<String>,
}

/// Device info
#[derive(Deserialize)]
pub struct DeviceInfo {
    device_id: String,
    display_name: Option<String>,
    created_at: i64,
    last_seen: Option<i64>,
}

impl RelayClient {
    /// Create a new relay client
    pub fn new(base_url: &str, auth_token: &str, device_id: [u8; 16]) -> Self {
        Self {
            base_url: base_url.to_string(),
            auth_token: auth_token.to_string(),
            device_id,
            client: reqwest::Client::new(),
        }
    }

    /// Create a relay client without device ID (for legacy compatibility)
    pub fn new_legacy(base_url: &str, auth_token: &str) -> Self {
        Self::new(base_url, auth_token, [0u8; 16])
    }

    /// Get device ID as hex string
    fn device_id_hex(&self) -> String {
        hex::encode(self.device_id)
    }

    /// Upload an encrypted diff to the relay (legacy method)
    pub async fn upload_diff(&self, payload: EncryptedPayload, root: &StateRoot) -> Result<()> {
        let url = format!("{}/api/v1/diffs", self.base_url);

        let request = UploadRequest {
            ciphertext: BASE64.encode(&payload.ciphertext),
            nonce: hex::encode(payload.nonce),
            signature: hex::encode(payload.signature),
            public_key: hex::encode(payload.public_key),
            root: root.hash_hex(),
        };

        let response = self
            .client
            .post(&url)
            .header("X-Device-ID", self.device_id_hex())
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&request)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to send upload request: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Upload failed with status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Push a signed diff to a specific recipient
    ///
    /// Per CP-013: Uses POST /push with X-Device-ID and X-Recipient-ID headers
    pub async fn push_diff(&self, signed_diff: &SignedDiff) -> Result<()> {
        let url = format!("{}/api/v1/push", self.base_url);

        #[derive(Serialize)]
        struct PushRequest {
            ciphertext: String,
            nonce: String,
            signature: String,
            public_key: String,
            outer_signature: String,
            sender_public_key: String,
            sequence: u64,
        }

        let request = PushRequest {
            ciphertext: BASE64.encode(&signed_diff.encrypted_diff.ciphertext),
            nonce: hex::encode(signed_diff.encrypted_diff.nonce),
            signature: hex::encode(signed_diff.encrypted_diff.signature),
            public_key: hex::encode(signed_diff.encrypted_diff.public_key),
            outer_signature: hex::encode(signed_diff.signature),
            sender_public_key: hex::encode(signed_diff.sender_public_key),
            sequence: signed_diff.sequence,
        };

        let response = self
            .client
            .post(&url)
            .header("X-Device-ID", hex::encode(signed_diff.sender_device_id))
            .header("X-Recipient-ID", hex::encode(signed_diff.target_device_id))
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&request)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to push diff: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Push failed with status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Pull diffs since a given sequence number
    ///
    /// Per CP-013: Uses GET /pull?since=N
    pub async fn pull_since(&self, since_seq: u64) -> Result<Vec<SignedDiff>> {
        let url = format!("{}/api/v1/pull?since={}", self.base_url, since_seq);

        let response = self
            .client
            .get(&url)
            .header("X-Device-ID", self.device_id_hex())
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to pull diffs: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Pull failed with status: {}",
                response.status()
            )));
        }

        #[derive(Deserialize)]
        struct PullResponse {
            diffs: Vec<PullDiff>,
        }

        #[derive(Deserialize)]
        struct PullDiff {
            ciphertext: String,
            nonce: String,
            signature: String,
            public_key: String,
            outer_signature: String,
            sender_public_key: String,
            sender_device_id: String,
            target_device_id: String,
            sequence: u64,
        }

        let data: PullResponse = response
            .json()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to parse pull response: {}", e)))?;

        let mut signed_diffs = Vec::new();
        for d in data.diffs {
            let ciphertext = BASE64
                .decode(&d.ciphertext)
                .map_err(|e| CPError::Sync(format!("Invalid base64: {}", e)))?;

            let nonce: [u8; 24] = hex::decode(&d.nonce)
                .map_err(|e| CPError::Sync(format!("Invalid nonce hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid nonce length".into()))?;

            let signature: [u8; 64] = hex::decode(&d.signature)
                .map_err(|e| CPError::Sync(format!("Invalid signature hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid signature length".into()))?;

            let public_key: [u8; 32] = hex::decode(&d.public_key)
                .map_err(|e| CPError::Sync(format!("Invalid public key hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid public key length".into()))?;

            let outer_signature: [u8; 64] = hex::decode(&d.outer_signature)
                .map_err(|e| CPError::Sync(format!("Invalid outer signature hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid outer signature length".into()))?;

            let sender_public_key: [u8; 32] = hex::decode(&d.sender_public_key)
                .map_err(|e| CPError::Sync(format!("Invalid sender public key hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid sender public key length".into()))?;

            let sender_device_id: [u8; 16] = hex::decode(&d.sender_device_id)
                .map_err(|e| CPError::Sync(format!("Invalid sender device ID hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid sender device ID length".into()))?;

            let target_device_id: [u8; 16] = hex::decode(&d.target_device_id)
                .map_err(|e| CPError::Sync(format!("Invalid target device ID hex: {}", e)))?
                .try_into()
                .map_err(|_| CPError::Sync("Invalid target device ID length".into()))?;

            signed_diffs.push(SignedDiff {
                encrypted_diff: EncryptedPayload {
                    ciphertext,
                    nonce,
                    signature,
                    public_key,
                },
                signature: outer_signature,
                sender_public_key,
                sender_device_id,
                target_device_id,
                sequence: d.sequence,
            });
        }

        Ok(signed_diffs)
    }

    /// Acknowledge receipt of diffs up to a given sequence
    ///
    /// Per CP-013: Uses DELETE /acknowledge with sequence parameter
    pub async fn acknowledge(&self, sequence: u64) -> Result<()> {
        let url = format!("{}/api/v1/acknowledge?seq={}", self.base_url, sequence);

        let response = self
            .client
            .delete(&url)
            .header("X-Device-ID", self.device_id_hex())
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to acknowledge: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Acknowledge failed with status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Get state roots from the relay
    /// 
    /// Note: The minimal relay server implementation currently returns ALL roots.
    /// Filtering by `since` would implementation on client side or server update.
    pub async fn get_roots(&self, _since: Option<[u8; 32]>) -> Result<Vec<String>> {
        let url = format!("{}/api/v1/roots", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to get roots: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Get roots failed with status: {}",
                response.status()
            )));
        }

        let roots: Vec<RootInfo> = response
            .json()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to parse roots: {}", e)))?;

        // Return just the hex strings for now, as recreating StateRoot requires more info
        // (The minimal relay doesn't return full StateRoot, just summary)
        Ok(roots.into_iter().map(|r| r.root).collect())
    }

    /// Get a specific diff by root hash
    pub async fn get_diff(&self, root_hex: &str) -> Result<EncryptedPayload> {
        let url = format!("{}/api/v1/diffs/{}", self.base_url, root_hex);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to get diff {}: {}", root_hex, e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Get diff failed with status: {}",
                response.status()
            )));
        }

        // Response is the same UploadRequest structure
        #[derive(Deserialize)]
        struct DownloadResponse {
            ciphertext: String,
            nonce: String,
            signature: String,
            public_key: String,
            // root: String, // Ignored
        }

        let data: DownloadResponse = response
            .json()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to parse diff response: {}", e)))?;

        // Decode fields
        let ciphertext = BASE64
            .decode(&data.ciphertext)
            .map_err(|e| CPError::Sync(format!("Invalid base64: {}", e)))?;

        let nonce_vec = hex::decode(&data.nonce)
            .map_err(|e| CPError::Sync(format!("Invalid nonce hex: {}", e)))?;
        let nonce: [u8; 24] = nonce_vec.try_into().map_err(|_| CPError::Sync("Invalid nonce length".into()))?;

        let sig_vec = hex::decode(&data.signature)
            .map_err(|e| CPError::Sync(format!("Invalid signature hex: {}", e)))?;
        let signature: [u8; 64] = sig_vec.try_into().map_err(|_| CPError::Sync("Invalid signature length".into()))?;

        let pk_vec = hex::decode(&data.public_key)
            .map_err(|e| CPError::Sync(format!("Invalid public key hex: {}", e)))?;
        let public_key: [u8; 32] = pk_vec.try_into().map_err(|_| CPError::Sync("Invalid public key length".into()))?;

        Ok(EncryptedPayload {
            ciphertext,
            nonce,
            signature,
            public_key,
        })
    }

    /// Check if the relay is reachable
    pub async fn health_check(&self) -> Result<bool> {
        let url = format!("{}/health", self.base_url);
        // Add timeout for health check
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap_or_else(|_| self.client.clone());

        match client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    // ============================================================================
    // Device Pairing Methods
    // ============================================================================

    /// Initiate device pairing - step 1
    pub async fn pair_init(&self, device_id: &str, public_key: &str, display_name: Option<&str>) -> Result<PairInitResponse> {
        let url = format!("{}/api/v1/pair/init", self.base_url);

        let request = PairInitRequest {
            device_id: device_id.to_string(),
            public_key: public_key.to_string(),
            display_name: display_name.map(|s| s.to_string()),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&request)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to initiate pairing: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Pairing init failed with status: {}",
                response.status()
            )));
        }

        response
            .json()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to parse pairing response: {}", e)))
    }

    /// Respond to pairing - step 2
    pub async fn pair_respond(&self, pairing_id: &str, device_id: &str, public_key: &str, display_name: Option<&str>) -> Result<()> {
        let url = format!("{}/api/v1/pair/respond", self.base_url);

        let request = PairRespondRequest {
            pairing_id: pairing_id.to_string(),
            device_id: device_id.to_string(),
            public_key: public_key.to_string(),
            display_name: display_name.map(|s| s.to_string()),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .json(&request)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to respond to pairing: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Pairing respond failed with status: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Confirm pairing - step 3
    pub async fn pair_confirm(&self, pairing_id: &str) -> Result<PairConfirmResponse> {
        let url = format!("{}/api/v1/pair/confirm", self.base_url);

        let request = PairConfirmRequest {
            pairing_id: pairing_id.to_string(),
        };

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("X-Device-ID", hex::encode(self.device_id))
            .json(&request)
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to confirm pairing: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "Pairing confirm failed with status: {}",
                response.status()
            )));
        }

        response
            .json()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to parse confirm response: {}", e)))
    }

    /// List paired devices
    pub async fn list_devices(&self) -> Result<Vec<DeviceInfo>> {
        let url = format!("{}/api/v1/devices", self.base_url);

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("X-Device-ID", hex::encode(self.device_id))
            .send()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to list devices: {}", e)))?;

        if !response.status().is_success() {
            return Err(CPError::Sync(format!(
                "List devices failed with status: {}",
                response.status()
            )));
        }

        response
            .json()
            .await
            .map_err(|e| CPError::Sync(format!("Failed to parse devices response: {}", e)))
    }
}
