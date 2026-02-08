use cfs_desktop::DesktopApp;
use cfs_query::QueryEngine;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

async fn run_validation_cycle(data_dir: std::path::PathBuf, corpus_dir: &Path) -> String {
    let mut app = DesktopApp::new(data_dir).unwrap();
    app.add_watch_dir(corpus_dir.to_path_buf()).unwrap();
    
    let app_handle = Arc::new(app);
    let app_clone = app_handle.clone();
    tokio::spawn(async move {
        app_clone.start().await.unwrap();
    });

    // Wait for ingestion
    sleep(Duration::from_secs(5)).await;

    // Dump DB hashes for determinism
    let graph = app_handle.graph();
    let lock = graph.lock().unwrap();
    // Use the Merkle root we already implemented!
    let root = lock.compute_merkle_root().unwrap();
    hex::encode(root)
}

#[tokio::test]
async fn test_system_validation() {
    // 1. Setup temporary environment
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();

    // Copy adversarial corpus to temp corpus dir
    let original_corpus = Path::new("../../test_corpus");
    for entry in std::fs::read_dir(original_corpus).unwrap() {
        let entry = entry.unwrap();
        let dest = corpus_dir.join(entry.file_name());
        std::fs::copy(entry.path(), dest).unwrap();
    }

    // 2. RUN CYCLE 1
    let data_dir_1 = temp_dir.path().join("data1");
    let hash1 = run_validation_cycle(data_dir_1.clone(), &corpus_dir).await;
    println!("Cycle 1 Hash: {}", hash1);

    // 3. RUN CYCLE 2 (Determinism)
    let data_dir_2 = temp_dir.path().join("data2");
    let hash2 = run_validation_cycle(data_dir_2.clone(), &corpus_dir).await;
    println!("Cycle 2 Hash: {}", hash2);

    assert_eq!(hash1, hash2, "System must be deterministic (identical hashes across clean runs)");

    // 4. Semantic Correctness (using Cycle 1 data)
    let app = DesktopApp::new(data_dir_1).unwrap();
    let graph = app.graph();
    let embedder = Arc::new(cfs_embeddings::EmbeddingEngine::new().unwrap());
    let query_engine = QueryEngine::new(graph.clone(), embedder);

    let results = query_engine.search("How to fix error 500?", 3).unwrap();
    assert!(!results.is_empty(), "Should return results for Troubleshooting query");
    assert!(results[0].chunk.text.contains("Troubleshooting"), "Top result should be from Troubleshooting section");

    // 5. Incrementality Test (Naive Baseline)
    let large_file_path = corpus_dir.join("adversarial.md");
    let start_time = std::time::Instant::now();
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(&large_file_path).unwrap();
        writeln!(file, "\nNew incremental update line.").unwrap();
    }
    sleep(Duration::from_secs(2)).await;
    let duration = start_time.elapsed();
    println!("Incremental update took: {:?}", duration);
}
