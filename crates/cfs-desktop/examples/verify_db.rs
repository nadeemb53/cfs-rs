use cfs_graph::GraphStore;
use std::path::PathBuf;

fn main() -> cfs_core::Result<()> {
    let db_path = "/Users/nadeem/dev/CFS/apps/macos/src-tauri/.cfs/graph.db";
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
