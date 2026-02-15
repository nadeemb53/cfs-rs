//! CP Relay Server binary

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging - default to INFO if RUST_LOG is not set
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(filter)
        .init();

    let addr = std::env::var("CP_RELAY_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let storage = std::env::var("CP_RELAY_STORAGE").unwrap_or_else(|_| "cp_relay.db".to_string());

    cp_relay_server::run(&addr, &storage).await
}
