# Canon Protocol

A deterministic, verifiable semantic state substrate for AI cognition and multi-agent reasoning.

---

## What problem are we solving?

AI agents today cannot verify each other's reasoning state, produce opaque context and hallucinated outputs, and have no shared semantic truth. Canon Protocol solves this by making state **deterministic, reproducible, and cryptographically verifiable**.

---

## Key Concepts

- **Canonicalization:** consistent byte-level normalization of all artifacts
- **Content-addressed state:** everything is hash-identified
- **Deterministic retrieval:** no nondeterministic floating point or host discrepancies
- **Verifiable execution traces:** replay proofs for agent state
- **Merkle roots for semantic state:** portable and auditable semantic snapshots

---

## Use Cases

- Trustless agent verification
- Collaborative AI knowledge networks
- Verifiable agent markets
- Auditable retrieval and provenance

---

## How It Works

Canon Protocol is composed of:

- **spec/** – protocol documents defining canonical behavior
- **crates/** – Rust implementation of core libraries
- **examples/** – illustrative workflows (not core deliverables yet)

---

## Core Principles

- **Deterministic**
  Identical inputs produce identical state roots, chunk IDs, and retrieval results. This is enforced via **SoftFloat** arithmetic and **Canonical Inference**.

- **Content-addressed**
  All state is identified by cryptographic hashes, enabling verification and deduplication.

- **Verifiable**
  Every state transition can be proven and replayed.

- **Local-first**
  All operations run locally without external dependencies.

---

## Architecture & Specifications

### Core Layer

- **[CP-001: Data Model](./spec/core/CP-001-data-model.md)** — Canonical data types (Document, Chunk, Embedding), UUIDv5 identity generation, and Merkle tree construction.
- **[CP-002: Storage Engine](./spec/core/CP-002-storage-engine.md)** — Hybrid persistence using SQLite for metadata and HNSW for transient vector indices.
- **[CP-003: Determinism](./spec/core/CP-003-determinism.md)** — Mathematical rules for SoftFloat arithmetic and Canonical Inference to guarantee bit-exact synchronization.

### Protocol Layer

- **[CP-010: Embedding](./spec/protocol/CP-010-embedding-protocol.md)** — Deterministic vector generation using quantization and provenance tracking.
- **[CP-011: Indexing](./spec/protocol/CP-011-indexing-protocol.md)** — Document ingestion, chunking strategies, and index maintenance.
- **[CP-012: Retrieval](./spec/protocol/CP-012-retrieval-protocol.md)** — Hybrid search (Semantic + Lexical), RRF fusion, and Integer Dot Product scoring.
- **[CP-013: Synchronization](./spec/protocol/CP-013-sync-protocol.md)** — Encrypted, blind synchronization using Merkle diffs and signatures.

### Application Layer

- **[CP-020: Intelligence Interface](./spec/application/CP-020-intelligence-interface.md)** — strict read-only contract for LLM integration.
- **[CP-021: Context Assembly](./spec/application/CP-021-context-assembly.md)** — Deterministic context window construction for RAG.

---

## Repository Structure

### Core Crates (`/crates`)

- `cp-core` — Canonical data models, hashing, cryptographic primitives
- `cp-parser` — Document parsing and chunking (PDF, Markdown, Text)
- `cp-embeddings` — Local embedding generation (CPU-only)
- `cp-graph` — SQLite + HNSW hybrid storage engine
- `cp-query` — Hybrid retrieval, RRF fusion, and context assembly
- `cp-inference-mobile` — Local LLM inference engine (`llama.cpp` + GGUF)
- `cp-sync` — Merkle tree diffing, encryption, and state convergence
- `cp-relay-client` — HTTP client for encrypted blob synchronization
- `cp-desktop` — Desktop-specific ingestion and watcher logic
- `cp-mobile` — C FFI for iOS/Android integration
- `cp-desktop-cli` — Command-line interface for graph inspection
- `cp-tests` — End-to-end and cross-platform validation tests

### Examples (`/examples`)

- `examples/macos` — macOS Tauri UI wrapper
- `examples/ios` — iOS SwiftUI application

### Relay Server (`/relay`)

- `relay/cp-relay-server` — Blind Axum-based encrypted blob storage

### Infrastructure & Tools

- `scripts/` — Build and deployment scripts (e.g., iOS cross-compilation)
- `test_corpus/` — Curated dataset for system validation and RAG testing
