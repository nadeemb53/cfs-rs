use cp_graph::GraphStore;
use std::path::PathBuf;

fn main() -> cp_core::Result<()> {
    let db_path = "/Users/nadeem/dev/CP/apps/macos/src-tauri/.cp/graph.db";
    let graph = GraphStore::open(db_path)?;
    let docs = graph.get_all_documents()?;
    
    println!("Document Count: {}", docs.len());
    for doc in docs {
        println!(" - {}", doc.path.display());
    }
    
    let stats = graph.stats()?;
    println!("Stats: Documents={}, Chunks={}, Embeddings={}", 
        stats.documents, stats.chunks, stats.embeddings);
        
    Ok(())
}
