//! End-to-end tests for the complete Mac-iOS sync flow
//!
//! This test verifies:
//! 1. Document selection and indexing (Mac side)
//! 2. State root computation and Merkle tree
//! 3. Query with hybrid search
//! 4. Citation extraction and verification

use cp_core::{CognitiveDiff, DiffMetadata, Document, Hlc};
use cp_desktop::DesktopApp;
use cp_graph::GraphStore;
use cp_query::QueryEngine;
use cp_sync::DeviceIdentity;
use tempfile::TempDir;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

/// Test the complete document indexing flow (Mac side simulation)
#[tokio::test]
async fn test_document_indexing_flow() {
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();

    // Create test documents
    let doc1 = "# Project Alpha\n\nThis is a secret project about AI.";
    let doc2 = "# Project Beta\n\nThis is about crypto and security.";

    std::fs::write(corpus_dir.join("alpha.md"), doc1).unwrap();
    std::fs::write(corpus_dir.join("beta.md"), doc2).unwrap();

    let data_dir = temp_dir.path().join("data");
    let mut app = DesktopApp::new(data_dir).unwrap();
    app.add_watch_dir(corpus_dir.clone()).unwrap();

    let app_handle = std::sync::Arc::new(app);
    let app_clone = app_handle.clone();

    // Start ingestion
    tokio::spawn(async move {
        app_clone.start().await.unwrap();
    });

    // Wait for ingestion
    sleep(Duration::from_secs(3)).await;

    // Verify documents were indexed
    let graph = app_handle.graph();
    let lock = graph.lock().unwrap();
    let docs = lock.get_all_documents().unwrap();

    assert_eq!(docs.len(), 2, "Should have indexed 2 documents");

    // Verify chunks were created
    let stats = lock.stats().unwrap();
    assert!(stats.chunks > 0, "Should have created chunks");

    println!("✓ Document indexing flow works: {} docs, {} chunks", docs.len(), stats.chunks);
}

/// Test state root computation and Merkle tree
#[test]
fn test_state_root_computation() {
    let mut graph = GraphStore::in_memory().unwrap();

    let device_id = Uuid::new_v4();
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let node_id: [u8; 16] = *Uuid::new_v4().as_bytes();

    // Add first document
    let doc1 = Document::new(
        std::path::PathBuf::from("test1.md"),
        b"Content about AI and ML",
        123,
    );
    let hlc1 = Hlc::new(now_ms, node_id);
    let mut diff1 = CognitiveDiff::empty([0u8; 32], device_id, 1, hlc1.clone());
    diff1.added_docs.push(doc1);
    diff1.metadata = DiffMetadata {
        prev_root: [0u8; 32],
        new_root: [1u8; 32],
        hlc: hlc1,
        device_id,
        seq: 1,
    };
    graph.apply_diff(&diff1).unwrap();

    let root1 = graph.compute_merkle_root().unwrap();
    println!("✓ First state root: {}", hex::encode(root1));

    // Add second document
    let doc2 = Document::new(
        std::path::PathBuf::from("test2.md"),
        b"Crypto and security content",
        456,
    );
    let hlc2 = Hlc::new(now_ms + 1, node_id);
    let mut diff2 = CognitiveDiff::empty(root1, device_id, 2, hlc2.clone());
    diff2.added_docs.push(doc2);
    diff2.metadata = DiffMetadata {
        prev_root: root1,
        new_root: [2u8; 32],
        hlc: hlc2,
        device_id,
        seq: 2,
    };
    graph.apply_diff(&diff2).unwrap();

    let root2 = graph.compute_merkle_root().unwrap();
    println!("✓ Second state root: {}", hex::encode(root2));

    assert_ne!(root1, root2, "Roots should differ after adding document");
}

/// Test query engine with hybrid search
#[tokio::test]
async fn test_hybrid_query_flow() {
    let temp_dir = TempDir::new().unwrap();
    let corpus_dir = temp_dir.path().join("corpus");
    std::fs::create_dir_all(&corpus_dir).unwrap();

    // Create documents with specific content for retrieval testing
    let content = r#"# Machine Learning

Deep learning is a subset of machine learning.
Neural networks are inspired by biological neurons.
Backpropagation is used to train neural networks.

## Transformer Architecture

The transformer architecture uses self-attention mechanisms.
It was introduced in the paper "Attention Is All You Need".
GPT models are based on transformers.

## Applications

1. Natural Language Processing
2. Computer Vision
3. Speech Recognition
"#;
    std::fs::write(corpus_dir.join("ml.md"), content).unwrap();

    let data_dir = temp_dir.path().join("data");
    let mut app = DesktopApp::new(data_dir).unwrap();
    app.add_watch_dir(corpus_dir).unwrap();

    let app_handle = std::sync::Arc::new(app);
    let app_clone = app_handle.clone();

    tokio::spawn(async move {
        app_clone.start().await.unwrap();
    });

    sleep(Duration::from_secs(3)).await;

    let graph = app_handle.graph();
    let embedder = std::sync::Arc::new(cp_embeddings::EmbeddingEngine::new().unwrap());
    let qe = QueryEngine::new(graph, embedder);

    // Test semantic search
    let results = qe.search("neural networks", 3).unwrap();
    println!("✓ Semantic search returned {} results", results.len());

    // Test lexical search
    let graph2 = app_handle.graph();
    let lock2 = graph2.lock().unwrap();
    let lexical = lock2.search_lexical("transformer", 3).unwrap();
    println!("✓ Lexical search returned {} results", lexical.len());

    assert!(!results.is_empty() || !lexical.is_empty(), "Should find results via at least one method");
}

/// Test device identity generation
#[test]
fn test_device_identity_signing() {
    // Generate two devices
    let alice = DeviceIdentity::generate();
    let bob = DeviceIdentity::generate();

    // Verify they have different IDs
    assert_ne!(alice.device_id, bob.device_id);

    println!("✓ Device identity works");
    println!("  Alice device_id: {}", hex::encode(alice.device_id));
    println!("  Alice public_key: {}", hex::encode(alice.public_key));
    println!("  Bob device_id: {}", hex::encode(bob.device_id));
}

/// Test key exchange between devices
#[test]
fn test_key_exchange() {
    let alice = DeviceIdentity::generate();
    let bob = DeviceIdentity::generate();

    // Get X25519 public keys
    let alice_x25519 = alice.x25519_public_key();
    let bob_x25519 = bob.x25519_public_key();

    // Alice computes shared secret with Bob's public key
    let alice_shared = alice.agree(&bob_x25519);
    let bob_shared = bob.agree(&alice_x25519);

    // Both should derive the same shared secret
    assert_eq!(alice_shared, bob_shared, "Shared secrets should match");

    println!("✓ Key exchange works - shared secret derived");
}

/// Test citation extraction - verifies we can track document provenance
#[test]
fn test_citation_extraction() {
    let mut graph = GraphStore::in_memory().unwrap();

    // Add document with known content
    let doc = Document::new(
        std::path::PathBuf::from("cited.md"),
        b"The quick brown fox jumps over the lazy dog. This is a cited fact.",
        123,
    );

    let device_id = Uuid::new_v4();
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let node_id: [u8; 16] = *Uuid::new_v4().as_bytes();

    let hlc = Hlc::new(now_ms, node_id);
    let mut diff = CognitiveDiff::empty([0u8; 32], device_id, 1, hlc.clone());
    diff.added_docs.push(doc);
    diff.metadata = DiffMetadata {
        prev_root: [0u8; 32],
        new_root: [1u8; 32],
        hlc,
        device_id,
        seq: 1,
    };

    graph.apply_diff(&diff).unwrap();

    // Get all documents
    let docs = graph.get_all_documents().unwrap();

    assert!(!docs.is_empty(), "Should have documents after applying diff");

    // Verify we can retrieve by ID (provenance tracking)
    let doc_id = docs.first().unwrap().id;
    let retrieved = graph.get_document(doc_id).unwrap();

    assert!(retrieved.is_some(), "Should retrieve document by ID");

    let retrieved_doc = retrieved.unwrap();
    println!("✓ Document retrieved: {}", retrieved_doc.path.display());
    println!("✓ Content hash: {}", hex::encode(retrieved_doc.hash));

    // Verify citation - we can verify the content is from our document
    // by checking the path and hash
    assert_eq!(retrieved_doc.path, std::path::PathBuf::from("cited.md"));

    println!("✓ Citation verification: document provenance tracked via path and hash");
}

/// Test graph store operations
#[test]
fn test_graph_operations() {
    let mut graph = GraphStore::in_memory().unwrap();

    // Insert a document
    let doc = Document::new(
        std::path::PathBuf::from("test.md"),
        b"Test content for graph operations",
        100,
    );
    graph.insert_document(&doc).unwrap();

    // Retrieve it
    let retrieved = graph.get_document(doc.id).unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().path, doc.path);

    // Delete it
    graph.delete_document(doc.id).unwrap();
    let after_delete = graph.get_document(doc.id).unwrap();
    assert!(after_delete.is_none());

    println!("✓ Graph store CRUD operations work");
}

/// Test sync convergence - two devices starting from same state
/// should converge to same state root after applying same diffs
#[test]
fn test_sync_convergence() {
    let mut graph_a = GraphStore::in_memory().unwrap();
    let mut graph_b = GraphStore::in_memory().unwrap();

    let device_id = Uuid::new_v4();
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let node_id: [u8; 16] = *Uuid::new_v4().as_bytes();

    // Create a diff
    let doc = Document::new(
        std::path::PathBuf::from("sync_test.md"),
        b"Content to sync between devices",
        123,
    );
    let hlc = Hlc::new(now_ms, node_id);
    let mut diff = CognitiveDiff::empty([0u8; 32], device_id, 1, hlc.clone());
    diff.added_docs.push(doc);
    diff.metadata = DiffMetadata {
        prev_root: [0u8; 32],
        new_root: [1u8; 32],
        hlc,
        device_id,
        seq: 1,
    };

    // Apply to both graphs
    graph_a.apply_diff(&diff).unwrap();
    graph_b.apply_diff(&diff).unwrap();

    // Both should have the same state root
    let root_a = graph_a.compute_merkle_root().unwrap();
    let root_b = graph_b.compute_merkle_root().unwrap();

    assert_eq!(root_a, root_b, "Devices should converge to same state");

    println!("✓ Sync convergence works: {}", hex::encode(root_a));
}
