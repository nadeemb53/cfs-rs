use cp_desktop::DesktopApp;
use cp_query::QueryEngine;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn test_lexical_vs_vector_retrieval() {
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();
    
    // Create a file with a very specific technical identifier
    let identifier = "ERR_CODE_0xDEADBEEF_42";
    let content = format!("# Technical Log\nSystem failed with error code: {}", identifier);
    std::fs::write(corpus_dir.join("tech_log.md"), &content).unwrap();

    let data_dir = temp_dir.path().join("data_retrieval");
    let mut app = DesktopApp::new(data_dir).unwrap();
    app.add_watch_dir(corpus_dir).unwrap();
    
    let app_handle = std::sync::Arc::new(app);
    let app_clone = app_handle.clone();
    
    tokio::spawn(async move {
        app_clone.start().await.unwrap();
    });

    sleep(Duration::from_secs(5)).await;

    let graph = app_handle.graph();
    let embedder = std::sync::Arc::new(cp_embeddings::EmbeddingEngine::new().unwrap());
    
    // 1. Vector Search (might miss exact match)
    let qe = QueryEngine::new(graph.clone(), embedder.clone());
    let vector_results = qe.search(identifier, 1).unwrap();
    println!("Vector Top Result: {:?}", vector_results.get(0).map(|r| &r.chunk.text));

    // 2. Lexical Search (should hit exact match)
    let lock = graph.lock().unwrap();
    let lexical_results = lock.search_lexical(identifier, 1).unwrap();
    println!("Lexical Results: {:?}", lexical_results);
    
    assert!(!lexical_results.is_empty(), "Lexical search must find the exact identifier");
}
