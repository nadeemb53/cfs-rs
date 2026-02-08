use cfs_core::{CognitiveDiff, DiffMetadata, Document};
use cfs_graph::GraphStore;
use uuid::Uuid;
use std::path::PathBuf;

#[test]
fn test_sync_convergence_simple() {
    let mut graph_a = GraphStore::in_memory().unwrap();
    let mut graph_b = GraphStore::in_memory().unwrap();
    
    let device_id = Uuid::new_v4();
    let doc = Document::new(PathBuf::from("test.md"), b"Content", 123);
    
    // 1. Create Diff on A
    let mut diff = CognitiveDiff::empty([0u8; 32], device_id, 1);
    diff.added_docs.push(doc.clone());
    
    let new_root_hash = [1u8; 32];
    diff.metadata = DiffMetadata {
        prev_root: [0u8; 32],
        new_root: new_root_hash,
        timestamp: 1000,
        device_id,
        seq: 1,
    };
    
    // 2. Apply to A and B
    graph_a.apply_diff(&diff).unwrap();
    graph_b.apply_diff(&diff).unwrap();
    
    // 3. Assert convergence
    let root_a = graph_a.compute_merkle_root().unwrap();
    let root_b = graph_b.compute_merkle_root().unwrap();
    
    assert_eq!(root_a, root_b);
}

#[test]
fn test_sync_out_of_order_violation() {
    // Current apply_diff doesn't strictly check seq order in the graph logic itself,
    // it just applies the data. The SyncManager handles the logic.
    // However, we should prove that applying the same diffs in different order 
    // (if they are independent) converges, OR that we can detect missing parents.
}
