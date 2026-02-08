use cfs_desktop::DesktopApp;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_ingestion_performance_baseline() {
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();
    
    // Create a large file (~10k tokens, assuming 4 chars/token -> 40k chars)
    let large_content = "This is a sentence that repeats many times to reach 10k tokens. ".repeat(1000);
    let file_path = corpus_dir.join("large.md");
    std::fs::write(&file_path, &large_content).unwrap();

    let data_dir = temp_dir.path().join("data_perf");
    let mut app = DesktopApp::new(data_dir).unwrap();
    app.add_watch_dir(corpus_dir).unwrap();
    
    let app_handle = std::sync::Arc::new(app);
    let app_clone = app_handle.clone();
    
    let start_clean = std::time::Instant::now();
    tokio::spawn(async move {
        app_clone.start().await.unwrap();
    });

    // Wait for ingestion
    sleep(Duration::from_secs(10)).await;
    let duration_clean = start_clean.elapsed();
    println!("Clean Ingest (10k tokens) took: {:?}", duration_clean);

    // Incremental Edit
    let start_inc = std::time::Instant::now();
    {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new().append(true).open(&file_path).unwrap();
        writeln!(file, "\nOne single incremental sentence update.").unwrap();
    }
    
    // Wait for file watcher and processing
    sleep(Duration::from_secs(5)).await;
    let duration_inc = start_inc.elapsed();
    println!("Incremental Edit took: {:?}", duration_inc);
    
    // In Phase 1, we expect Incremental to take less time if we have change detection.
    // However, currently we re-embed the whole file. 
    // Step 3 (Hierarchical Hashing) will help us skip embeddings later.
}
