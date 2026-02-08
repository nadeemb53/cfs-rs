//! CFS Relay Client - HTTP client for sync operations
//!
//! Provides a strongly-typed client for the minimal relay server.
//! Handles encryption, serialization, and hex/base64 encoding.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use cfs_core::{CfsError, Result, StateRoot};
use cfs_sync::EncryptedPayload;
use serde::{Deserialize, Serialize};

/// Client for the CFS relay server
pub struct RelayClient {
    base_url: String,
    _auth_token: String,
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

impl RelayClient {
    /// Create a new relay client
    pub fn new(base_url: &str, auth_token: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
            _auth_token: auth_token.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Upload an encrypted diff to the relay
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
            .json(&request)
            .send()
            .await
            .map_err(|e| CfsError::Sync(format!("Failed to send upload request: {}", e)))?;

        if !response.status().is_success() {
            return Err(CfsError::Sync(format!(
                "Upload failed with status: {}",
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
            .map_err(|e| CfsError::Sync(format!("Failed to get roots: {}", e)))?;

        if !response.status().is_success() {
            return Err(CfsError::Sync(format!(
                "Get roots failed with status: {}",
                response.status()
            )));
        }

        let roots: Vec<RootInfo> = response
            .json()
            .await
            .map_err(|e| CfsError::Sync(format!("Failed to parse roots: {}", e)))?;

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
            .map_err(|e| CfsError::Sync(format!("Failed to get diff {}: {}", root_hex, e)))?;

        if !response.status().is_success() {
            return Err(CfsError::Sync(format!(
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
            .map_err(|e| CfsError::Sync(format!("Failed to parse diff response: {}", e)))?;

        // Decode fields
        let ciphertext = BASE64
            .decode(&data.ciphertext)
            .map_err(|e| CfsError::Sync(format!("Invalid base64: {}", e)))?;

        let nonce_vec = hex::decode(&data.nonce)
            .map_err(|e| CfsError::Sync(format!("Invalid nonce hex: {}", e)))?;
        let nonce: [u8; 24] = nonce_vec.try_into().map_err(|_| CfsError::Sync("Invalid nonce length".into()))?;

        let sig_vec = hex::decode(&data.signature)
            .map_err(|e| CfsError::Sync(format!("Invalid signature hex: {}", e)))?;
        let signature: [u8; 64] = sig_vec.try_into().map_err(|_| CfsError::Sync("Invalid signature length".into()))?;

        let pk_vec = hex::decode(&data.public_key)
            .map_err(|e| CfsError::Sync(format!("Invalid public key hex: {}", e)))?;
        let public_key: [u8; 32] = pk_vec.try_into().map_err(|_| CfsError::Sync("Invalid public key length".into()))?;

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
}
