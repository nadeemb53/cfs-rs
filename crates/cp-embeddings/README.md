# CP Embeddings

Local embedding generation using Candle (pure Rust ML framework).

## Overview

This crate provides the **Fast Path** per CP-010:
- Hardware-accelerated embedding generation using Candle
- Downloads model on first use to `~/.cp/models/`
- Uses MiniLM-L6-v2 (384 dimensions)

## Usage

```rust
use cp_embeddings::EmbeddingEngine;

let engine = EmbeddingEngine::new()?;
let embedding = engine.embed("Hello, world!")?;
```
