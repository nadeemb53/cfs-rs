//! CFS Relay Server - Encrypted blob relay
//!
//! A minimal server that stores and serves encrypted diffs.
//! The server never sees plaintext - it only stores encrypted blobs.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{get, post},
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

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
}

/// Upload request
#[derive(Deserialize, Serialize)]
pub struct UploadRequest {
    /// Base64-encoded ciphertext
    pub ciphertext: String,
    /// Hex-encoded nonce
    pub nonce: String,
    /// Hex-encoded signature
    pub signature: String,
    /// Hex-encoded public key
    pub public_key: String,
    /// Hex-encoded state root
    pub root: String,
}

/// Root info response
#[derive(Serialize)]
pub struct RootInfo {
    pub root: String,
    pub timestamp: i64,
    pub size: usize,
}

/// Health check endpoint
async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Upload a diff
async fn upload_diff(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UploadRequest>,
) -> Result<StatusCode, StatusCode> {
    info!("POST /api/v1/diffs - Root: {}", req.root);
    state
        .storage
        .store(
            &req.root,
            &req.ciphertext,
            &req.nonce,
            &req.signature,
            &req.public_key,
        )
        .map_err(|e| {
            tracing::error!("Failed to store diff: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(StatusCode::CREATED)
}

/// List state roots
async fn list_roots(State(state): State<Arc<AppState>>) -> Result<Json<Vec<RootInfo>>, StatusCode> {
    info!("GET /api/v1/roots");
    let roots = state
        .storage
        .list_roots()
        .map_err(|e| {
            tracing::error!("Failed to list roots: {}", e);
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Json(roots))
}

/// Get a specific diff
async fn get_diff(
    State(state): State<Arc<AppState>>,
    Path(root): Path<String>,
) -> Result<Json<UploadRequest>, StatusCode> {
    info!("GET /api/v1/diffs/{}", root);
    state
        .storage
        .get(&root)
        .map(Json)
        .map_err(|e| {
            tracing::error!("Diff not found {}: {}", root, e);
            StatusCode::NOT_FOUND
        })
}

/// Create the router
pub fn create_router(storage_path: &str) -> Router {
    let storage = Storage::open(storage_path).expect("Failed to open storage");

    let state = Arc::new(AppState { storage });

    Router::new()
        .route("/health", get(health))
        .route("/api/v1/diffs", post(upload_diff))
        .route("/api/v1/roots", get(list_roots))
        .route("/api/v1/diffs/:root", get(get_diff))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Run the server
pub async fn run(addr: &str, storage_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_router(storage_path);

    info!("Starting CFS relay server on {}", addr);

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
