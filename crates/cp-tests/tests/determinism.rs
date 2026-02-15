use cp_desktop::DesktopApp;
use std::path::PathBuf;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

async fn run_and_get_hash(data_dir: PathBuf, corpus_dir: PathBuf) -> String {
    let mut app = DesktopApp::new(data_dir).unwrap();
    app.add_watch_dir(corpus_dir).unwrap();
    
    let app_handle = std::sync::Arc::new(app);
    let app_clone = app_handle.clone();
    
    tokio::spawn(async move {
        app_clone.start().await.unwrap();
    });

    // Wait for ingestion completion (naive wait for now)
    sleep(Duration::from_secs(5)).await;

    let graph = app_handle.graph();
    let lock = graph.lock().unwrap();
    let root = lock.compute_merkle_root().unwrap();
    hex::encode(root)
}

#[tokio::test]
async fn test_clean_rebuild_determinism() {
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();
    
    // Create a stable corpus
    std::fs::write(corpus_dir.join("test1.md"), "# Test 1\nContent A").unwrap();
    std::fs::write(corpus_dir.join("test2.md"), "# Test 2\nContent B").unwrap();

    let data_dir_a = temp_dir.path().join("data_a");
    let hash_a = run_and_get_hash(data_dir_a, corpus_dir.clone()).await;

    let data_dir_b = temp_dir.path().join("data_b");
    let hash_b = run_and_get_hash(data_dir_b, corpus_dir).await;

    assert_eq!(hash_a, hash_b, "Hash A and B must be identical for clean rebuilds");
    println!("Determinism verified: {}", hash_a);
}

#[tokio::test]
async fn test_restart_determinism() {
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();
    std::fs::write(corpus_dir.join("test1.md"), "# Test 1\nContent A").unwrap();

    let data_dir = temp_dir.path().join("data_restart");
    
    // 1. Initial ingest
    let hash1 = run_and_get_hash(data_dir.clone(), corpus_dir.clone()).await;
    
    // 2. Restart and re-check hash
    let mut app = DesktopApp::new(data_dir).unwrap();
    let graph = app.graph();
    let lock = graph.lock().unwrap();
    let hash2 = hex::encode(lock.compute_merkle_root().unwrap());

    assert_eq!(hash1, hash2, "Hash must be stable across restarts");
}
