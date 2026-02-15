# Canon Protocol Specification

> **Version**: 1.0.0
> **Authors**: Nadeem Bhati
> **Last Updated**: February 2026

## Abstract

Canon Protocol is a deterministic, verifiable semantic state substrate for AI cognition and multi-agent reasoning. It provides a verifiable substrate where AI agents can share state, verify each other's reasoning, and collaborate without centralized trust.

This specification defines the core data models, protocols, and interfaces enabling interoperability between implementations with formal guarantees about system behavior.

## Motivation

The proliferation of AI systems has introduced fundamental trust problems:

1. **Opaque Reasoning**: AI agents cannot verify each other's reasoning state
2. **Hallucinated Outputs**: No shared semantic truth between agents
3. **Hidden Mutation**: AI systems may produce inconsistent outputs without accountability
4. **Non-Determinism**: Identical inputs may produce different outputs due to floating-point discrepancies

Canon Protocol addresses these issues by establishing a **substrate-first architecture** where:

- All semantic operations are deterministic and reproducible
- State is cryptographically committed via Merkle trees
- Intelligence modules operate as read-only lenses on the substrate
- Synchronization is encrypted and verifiable

## Design Philosophy

### Substrate First, Intelligence Second

Canon Protocol inverts the traditional AI paradigm:

| Traditional AI | Canon Protocol |
|----------------|----------------|
| Opaque reasoning | Verifiable execution traces |
| Hidden state | Content-addressed state |
| Floating-point nondeterminism | SoftFloat arithmetic |
| Centralized trust | Cryptographic verification |

### Core Principles

1. **Deterministic**: Identical inputs always produce identical outputs—chunks, hashes, and retrieval results. This is enforced via **SoftFloat** arithmetic and **Canonical Inference**.
2. **Content-Addressed**: Everything is hash-identified, enabling verification and deduplication.
3. **Verifiable**: Every state transition can be proven and replayed.
4. **Inspectable**: All internal state is visible and debuggable.
5. **Mutable but Stable**: Changes trigger incremental updates without full reprocessing.
6. **Local-First**: All processing runs locally without external dependencies.

## Specification Index

### Core Specifications

These specifications define the fundamental data models and cryptographic primitives:

| Spec | Title | Description |
|------|-------|-------------|
| [CP-001](./core/CP-001-data-model.md) | Data Model & Identity | Canonical data types, hashing, Merkle trees |
| [CP-002](./core/CP-002-storage-engine.md) | Storage Engine | Hybrid SQLite + HNSW architecture |
| [CP-003](./core/CP-003-determinism.md) | Determinism & Math | SoftFloat math, canonical inference, transient indices |

### Protocol Specifications

These specifications define the runtime behavior of Canon Protocol components:

| Spec | Title | Description |
|------|-------|-------------|
| [CP-010](./protocol/CP-010-embedding-protocol.md) | Embedding Protocol | Vector generation, model provenance, storage format |
| [CP-011](./protocol/CP-011-indexing-protocol.md) | Indexing Protocol | Document ingestion, chunking, index construction |
| [CP-012](./protocol/CP-012-retrieval-protocol.md) | Retrieval Protocol | Hybrid search, RRF fusion, result ranking |
| [CP-013](./protocol/CP-013-sync-protocol.md) | Synchronization Protocol | Merkle verification, diff generation, encrypted relay |

### Application Specifications

These specifications define the interfaces between Canon Protocol and external systems:

| Spec | Title | Description |
|------|-------|-------------|
| [CP-020](./application/CP-020-intelligence-interface.md) | Intelligence Interface | LLM integration contract, read-only guarantees |
| [CP-021](./application/CP-021-context-assembly.md) | Context Assembly | Deterministic context window construction |

## Terminology

| Term | Definition |
|------|------------|
| **Substrate** | The complete semantic state comprising all documents, chunks, embeddings, and edges |
| **State Root** | Cryptographic commitment (BLAKE3 hash) to the entire substrate state |
| **Canonicalization** | Consistent byte-level normalization of all artifacts |
| **Cognitive Diff** | Atomic unit of state change, containing added/modified/removed entities |
| **Intelligence Module** | Read-only component that interprets substrate content (e.g., LLM) |
| **Hybrid Search** | Combined semantic (vector) and lexical (keyword) retrieval |
| **RRF** | Reciprocal Rank Fusion—algorithm for combining multiple ranked result sets |
| **Relay Server** | Blind intermediary for encrypted state synchronization |

## Versioning

Canon Protocol specifications follow [Semantic Versioning 2.0.0](https://semver.org/):

- **MAJOR**: Breaking changes to data models or protocols
- **MINOR**: Backwards-compatible feature additions
- **PATCH**: Backwards-compatible bug fixes and clarifications

## Implementation Requirements

Conformant Canon Protocol implementations MUST:

1. Produce identical outputs for identical inputs (determinism)
2. Support all core data types as specified in CP-001
3. Implement BLAKE3 hashing as specified in CP-002
4. Support hybrid retrieval as specified in CP-012
5. Enforce intelligence module constraints as specified in CP-020

## Security Considerations

Canon Protocol is designed with the following security properties:

1. **Data Sovereignty**: All processing is local; no data leaves the device without explicit user action
2. **End-to-End Encryption**: Synchronization uses XChaCha20-Poly1305 with device-specific keys
3. **Cryptographic Integrity**: State roots provide tamper-evidence via BLAKE3 Merkle trees
4. **Relay Blindness**: Synchronization servers cannot decrypt or inspect user content
5. **Signature Verification**: Ed25519 signatures authenticate state transitions

## Contributing

Contributions to these specifications are welcome. Please submit issues and pull requests to the Canon Protocol repository.

## License

These specifications are released under the Apache 2.0 License.
