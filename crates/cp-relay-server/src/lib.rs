//! CP Relay Server - Encrypted blob relay
//!
//! A minimal server that stores and serves encrypted diffs.
//! The server never sees plaintext - it only stores encrypted blobs.
//!
//! Per CP-013: Implements the blind relay protocol with:
//! - POST /push - Upload a signed diff
//! - GET /pull - Pull diffs since a sequence
//! - DELETE /acknowledge - Acknowledge receipt of diffs

use axum::{
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing::info;

mod storage;

use storage::Storage;

/// Application state
pub struct AppState {
    storage: Storage,
}

/// Push response
#[derive(Serialize)]
pub struct PushResponse {
    pub success: bool,
    pub sequence: u64,
}

/// Pull query parameters
#[derive(Deserialize)]
pub struct PullQuery {
    since: u64,
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
    // Extract headers per CP-013
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

    // Store the diff and get sequence number
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
    Query(query): Query<PullQuery>,
    headers: axum::http::HeaderMap,
) -> Result<Json<PullResponse>, StatusCode> {
    // Extract recipient device ID from header per CP-013
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

/// Create the router
pub fn create_router(storage_path: &str) -> Router {
    let storage = Storage::open(storage_path).expect("Failed to open storage");

    let state = Arc::new(AppState { storage });

    Router::new()
        .route("/health", get(health))
        .route("/push", post(push_diff))
        .route("/pull", get(pull_diffs))
        .route("/acknowledge", delete(acknowledge_diffs))
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
