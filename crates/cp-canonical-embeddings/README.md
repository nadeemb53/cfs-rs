# CP Canonical Embeddings

Deterministic embedding generation for Canon Protocol (CP).

## Overview

Per CP-010, this crate provides:
- **Canonical Path**: Software Float (SoftFloat) for deterministic L2 normalization
- Full embedding pipeline: Tokenization → Transformer → Pooling → Normalization → Quantization

## Model

Uses MiniLM-L6-v2 (384 dimensions) with:
- BLAKE3-based model hashing for provenance
- i16 quantization for cross-platform consistency
- SoftFloat for bit-exact results

## Usage

```rust
use cp_canonical_embeddings::{embed_text, Embedding};

let embedding = embed_text("Hello, world!")?;
println!("Embedding ID: {:?}", embedding.id());
```
