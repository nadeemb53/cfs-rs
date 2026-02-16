# CP Query

RAG query engine for Canon Protocol (CP).

## Overview

CP Query provides:
- Context assembly for LLM prompts
- Retrieval augmented generation utilities
- Query processing and ranking

## Usage

```rust
use cp_query::QueryEngine;

let engine = QueryEngine::new(graph_store)?;
let context = engine.assemble_context(query, max_tokens)?;
```
