# CP Graph

Knowledge graph and vector store for Canon Protocol (CP).

## Overview

CP Graph provides:
- SQLite-based storage for documents, chunks, embeddings
- Vector indexing with HNSW
- Merkle tree root computation
- SQL migrations

## Usage

```rust
use cp_graph::GraphStore;

let store = GraphStore::new(db_path)?;
let docs = store.get_all_documents()?;
```
