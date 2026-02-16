# CP Core

Core data models and traits for Canon Protocol (CP).

## Overview

CP Core provides the foundational data types used across the CP ecosystem:
- Document, Chunk, and Embedding structures
- ID generation (BLAKE3-based content-addressing)
- HLC (Hybrid Logical Clock) implementation
- Edge and state management
- Text normalization per CP-003

## Usage

```rust
use cp_core::{Document, Chunk, Id};

let doc = Document::new(path, content)?;
```
