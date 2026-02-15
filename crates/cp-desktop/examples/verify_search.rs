use cp_graph::GraphStore;
use cp_query::QueryEngine;
use cp_embeddings::EmbeddingEngine;
use std::sync::{Arc, Mutex};

fn main() -> cp_core::Result<()> {
    let db_path = "/Users/nadeem/dev/CP/apps/macos/src-tauri/.cp/graph.db";
    let graph = GraphStore::open(db_path)?;
    let graph_arc = Arc::new(Mutex::new(graph));
    let embedder = Arc::new(EmbeddingEngine::new()?);
    let qe = QueryEngine::new(graph_arc, embedder);
    
    let query = "ethereum";
    println!("Querying: '{}'", query);
    
    let results = qe.search(query, 5)?;
    println!("Results found: {}", results.len());
    for (i, res) in results.iter().enumerate() {
        println!("  {}. [Score: {:.4}] {} -> {}", 
            i + 1, res.score, res.doc_path, res.chunk.text.trim());
    }
    
    Ok(())
}
