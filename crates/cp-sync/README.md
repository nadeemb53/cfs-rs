# CP Sync

Merkle tree, cognitive diffs, and cryptography for Canon Protocol (CP).

## Overview

CP Sync provides:
- Merkle tree implementation for state roots
- Cognitive diff algorithms
- Cryptographic signing and verification
- State chain management

## Usage

```rust
use cp_sync::{MerkleTree, StateChain};

let tree = MerkleTree::build(&documents)?;
let root = tree.root();
```
