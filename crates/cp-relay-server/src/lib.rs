//! CP Relay Server - Encrypted blob relay
//!
//! A minimal server that stores and serves encrypted diffs.
//! The server never sees plaintext - it only stores encrypted blobs.
//!
//! Per CP-013: Implements the blind relay protocol with:
//! - POST /push - Upload a signed diff
//! - GET /pull - Pull diffs since a sequence
//! - DELETE /acknowledge - Acknowledge receipt of diffs
//! - POST /api/v1/pair/* - Device pairing endpoints
//! - GET /api/v1/devices - List paired devices

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;
use uuid::Uuid;

mod storage;

use storage::Storage;

/// Application state
pub struct AppState {
    storage: Storage,
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Push response
#[derive(Serialize)]
pub struct PushResponse {
    pub success: bool,
    pub sequence: u64,
}

/// Pull response
#[derive(Serialize)]
pub struct PullResponse {
    pub diffs: Vec<SignedDiffJson>,
}

/// Acknowledge request
#[derive(Deserialize)]
pub struct AcknowledgeRequest {
    pub sequence: u64,
}

/// Acknowledge response
#[derive(Serialize)]
pub struct AcknowledgeResponse {
    pub success: bool,
}

/// Signed diff JSON representation for API
#[derive(Serialize, Deserialize)]
pub struct SignedDiffJson {
    pub sequence: u64,
    pub sender_device_id: String,
    pub target_device_id: String,
    pub ciphertext: String,
    pub nonce: String,
    pub signature: String,
    pub sender_public_key: String,
    pub timestamp: i64,
}

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
}

/// Pairing initiation request
#[derive(Deserialize)]
pub struct PairInitRequest {
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
}

/// Pairing initiation response
#[derive(Serialize)]
pub struct PairInitResponse {
    pub pairing_id: String,
    pub pairing_code: String,
    pub expires_at: i64,
}

/// Pairing response request
#[derive(Deserialize)]
pub struct PairRespondRequest {
    pub pairing_id: String,
    pub device_id: String,
    pub public_key: String,
    pub display_name: Option<String>,
}

/// Pairing confirmation request
#[derive(Deserialize)]
pub struct PairConfirmRequest {
    pub pairing_id: String,
}

/// Pairing confirmation response
#[derive(Serialize)]
pub struct PairConfirmResponse {
    pub success: bool,
    pub peer_device_id: String,
    pub peer_display_name: Option<String>,
}

/// Device info response
#[derive(Serialize)]
pub struct DeviceInfo {
    pub device_id: String,
    pub display_name: Option<String>,
    pub created_at: i64,
    pub last_seen: Option<i64>,
}

// ============================================================================
// HTTP Handlers
// ============================================================================

/// Health check endpoint
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Push a diff to the relay
async fn push_diff(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Json<PushResponse>, StatusCode> {
    let sender_id = headers
        .get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();

    let recipient_id = headers
        .get("x-recipient-id")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();

    info!("POST /push - From: {} -> To: {}", sender_id, recipient_id);

    let (sequence, _timestamp) = state
        .storage
        .store_diff(&sender_id, &recipient_id, &body)
        .map_err(|e| {
            tracing::error!("Failed to store diff: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(PushResponse {
        success: true,
        sequence,
    }))
}

/// Pull diffs since a sequence number
async fn pull_diffs(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(query): axum::extract::Query<PullQuery>,
    headers: axum::http::HeaderMap,
) -> Result<Json<PullResponse>, StatusCode> {
    let recipient_id = headers
        .get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();

    info!("GET /pull - Recipient: {}, since: {}", recipient_id, query.since);

    let diffs = state
        .storage
        .get_diffs_since(&recipient_id, query.since)
        .map_err(|e| {
            tracing::error!("Failed to get diffs: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(PullResponse { diffs }))
}

/// Pull query parameters
#[derive(Deserialize)]
pub struct PullQuery {
    since: u64,
}

/// Acknowledge receipt of diffs
async fn acknowledge_diffs(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    Json(req): Json<AcknowledgeRequest>,
) -> Result<Json<AcknowledgeResponse>, StatusCode> {
    let device_id = headers
        .get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();

    info!("DELETE /acknowledge - Device: {}, sequence: {}", device_id, req.sequence);

    state
        .storage
        .acknowledge_diffs(&device_id, req.sequence)
        .map_err(|e| {
            tracing::error!("Failed to acknowledge diffs: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(AcknowledgeResponse { success: true }))
}

/// Generate 6-character pairing code
fn generate_pairing_code() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    (0..6)
        .map(|_| {
            let idx = rng.gen_range(0..36);
            if idx < 10 {
                (b'0' + idx) as char
            } else {
                (b'A' + idx - 10) as char
            }
        })
        .collect()
}

/// Initiate device pairing - step 1
async fn pair_init(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PairInitRequest>,
) -> Result<Json<PairInitResponse>, StatusCode> {
    let pairing_id = Uuid::new_v4().to_string();
    let pairing_code = generate_pairing_code();

    let expires_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
        + 300; // 5 minutes

    // Register the device first
    state
        .storage
        .register_device(
            &req.device_id,
            &req.public_key,
            req.display_name.as_deref().unwrap_or(""),
        )
        .map_err(|e| {
            tracing::error!("Failed to register device: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Store pending pairing
    state
        .storage
        .store_pending_pairing(
            &pairing_id,
            &req.device_id,
            &req.public_key,
            req.display_name.as_deref(),
            expires_at,
        )
        .map_err(|e| {
            tracing::error!("Failed to store pending pairing: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    info!("Pairing initiated: {} -> {}", pairing_id, req.device_id);

    Ok(Json(PairInitResponse {
        pairing_id,
        pairing_code,
        expires_at,
    }))
}

/// Respond to pairing - step 2
async fn pair_respond(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PairRespondRequest>,
) -> Result<Json<AcknowledgeResponse>, StatusCode> {
    // Register the responder device
    state
        .storage
        .register_device(
            &req.device_id,
            &req.public_key,
            req.display_name.as_deref().unwrap_or(""),
        )
        .map_err(|e| {
            tracing::error!("Failed to register device: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Update pending pairing with responder device
    state
        .storage
        .update_pending_pairing(
            &req.pairing_id,
            &req.device_id,
            &req.public_key,
            req.display_name.as_deref(),
        )
        .map_err(|e| {
            tracing::error!("Failed to update pending pairing: {}", e);
            StatusCode::NOT_FOUND
        })?;

    info!("Pairing responded: {}", req.pairing_id);

    Ok(Json(AcknowledgeResponse { success: true }))
}

/// Confirm pairing - step 3
async fn pair_confirm(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PairConfirmRequest>,
) -> Result<Json<PairConfirmResponse>, StatusCode> {
    // Get pending pairing
    let pending = state
        .storage
        .get_pending_pairing(&req.pairing_id)
        .map_err(|_| StatusCode::NOT_FOUND)?
        .ok_or(StatusCode::NOT_FOUND)?;

    // Verify both devices are present
    if pending.initiator_device_id.is_empty() || pending.responder_device_id.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Compute shared secret hash
    let combined = format!(
        "{}:{}",
        pending.initiator_public_key, pending.responder_public_key
    );
    let shared_hash = blake3::hash(combined.as_bytes());
    let shared_hash_hex = hex::encode(shared_hash.as_bytes());

    // Create pairing
    state
        .storage
        .create_pairing(
            &pending.initiator_device_id,
            &pending.responder_device_id,
            &shared_hash_hex,
        )
        .map_err(|e| {
            tracing::error!("Failed to create pairing: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    // Delete pending pairing
    state
        .storage
        .delete_pending_pairing(&req.pairing_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    info!(
        "Pairing confirmed: {} <-> {}",
        pending.initiator_device_id, pending.responder_device_id
    );

    Ok(Json(PairConfirmResponse {
        success: true,
        peer_device_id: pending.responder_device_id,
        peer_display_name: pending.responder_display_name,
    }))
}

/// List paired devices
async fn list_devices(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
) -> Result<Json<Vec<DeviceInfo>>, StatusCode> {
    let device_id = headers
        .get("x-device-id")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::BAD_REQUEST)?
        .to_string();

    let pairings = state
        .storage
        .get_pairings(&device_id)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let mut devices = Vec::new();
    for pairing in pairings {
        let peer_id = if pairing.device_a == device_id {
            pairing.device_b
        } else {
            pairing.device_a
        };

        if let Ok(Some(device)) = state.storage.get_device(&peer_id) {
            devices.push(DeviceInfo {
                device_id: device.device_id,
                display_name: device.display_name,
                created_at: device.created_at,
                last_seen: device.last_seen,
            });
        }
    }

    Ok(Json(devices))
}

// ============================================================================
// Router Setup
// ============================================================================

/// Create the router
pub fn create_router(storage_path: &str) -> Router {
    let storage = Storage::open(storage_path).expect("Failed to open storage");

    let state = Arc::new(AppState { storage });

    Router::new()
        .route("/health", get(health))
        .route("/push", post(push_diff))
        .route("/pull", get(pull_diffs))
        .route("/acknowledge", delete(acknowledge_diffs))
        .route("/api/v1/pair/init", post(pair_init))
        .route("/api/v1/pair/respond", post(pair_respond))
        .route("/api/v1/pair/confirm", post(pair_confirm))
        .route("/api/v1/devices", get(list_devices))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Run the server
pub async fn run(addr: &str, storage_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router(storage_path);

    info!("Starting CP relay server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_router(":memory:");

        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
