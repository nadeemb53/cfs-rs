# Complete Test Coverage Scenarios for Canon Protocol

> Generated: 2026-02-16
> Purpose: Comprehensive test scenarios to achieve 100% test coverage

---

## 1. cp-core - Core Data Models

### 1.1 ID Generation (`crates/cp-core/src/id.rs`)

```rust
#[test]
fn test_generate_id_from_bytes() {
    // Valid input bytes
}

#[test]
fn test_generate_id_determinism_same_input() {
    // Same input produces same ID
}

#[test]
fn test_generate_id_different_inputs_produce_different_ids() {
    // Different inputs produce different IDs
}

#[test]
fn test_generate_id_empty_input() {
    // Edge case: empty input
}

#[test]
fn test_generate_id_known_test_vector() {
    // CP-001 test vector verification
}

#[test]
fn test_id_length_exact_16_bytes() {
    // Verify exactly 16 bytes
}

#[test]
fn test_id_display_format() {
    // UUID-style display formatting
}

#[test]
fn test_id_from_blake3_first_16_bytes() {
    // Verify BLAKE3 truncation
}
```

### 1.2 Document (`crates/cp-core/src/document.rs`)

```rust
#[test]
fn test_document_new() {
    // Create new document with valid input
}

#[test]
fn test_document_id_derivation_from_content_hash() {
    // document_id = generate_id(content_hash)
}

#[test]
fn test_document_path_id_derivation() {
    // path_id = generate_id(path.as_bytes())
}

#[test]
fn test_document_hierarchical_hash_computation() {
    // Sort chunks by sequence, BLAKE3 join
}

#[test]
fn test_document_serialization() {
    // CBOR serialization round-trip
}

#[test]
fn test_document_deserialization_invalid() {
    // Malformed data handling
}

#[test]
fn test_document_canonical_bytes() {
    // Verify to_canonical_bytes() format
}

#[test]
fn test_document_mime_type_detection_markdown() {
    // .md -> text/markdown
}

#[test]
fn test_document_mime_type_detection_text() {
    // .txt -> text/plain
}

#[test]
fn test_document_mime_type_detection_unknown() {
    // Unknown extension -> application/octet-stream
}

#[test]
fn test_document_size_bytes_calculation() {
    // Verify size matches content length
}

#[test]
fn test_document_mtime_from_filesystem() {
    // mtime extraction
}
```

### 1.3 Chunk (`crates/cp-core/src/chunk.rs`)

```rust
#[test]
fn test_chunk_new() {
    // Create chunk with valid input
}

#[test]
fn test_chunk_id_stable_across_rechunking() {
    // chunk_id = generate_id(document_id + sequence)
    // Should NOT include text content
}

#[test]
fn test_chunk_id_different_sequence_different_id() {
    // Same content, different sequence = different ID
}

#[test]
fn test_chunk_text_hash_computation() {
    // text_hash = BLAKE3(canonicalized_text)
}

#[test]
fn test_chunk_byte_offset_validation() {
    // Offset must be valid
}

#[test]
fn test_chunk_sequence_ordering() {
    // Sequence numbers are 0-indexed
}

#[test]
fn test_chunk_canonical_bytes_format() {
    // Verify serialization format
}

#[test]
fn test_chunk_overlap_semantics() {
    // Verify overlap handling (byte_offset/byte_length)
}

#[test]
fn test_chunk_text_validation_utf8() {
    // Only valid UTF-8 allowed
}

#[test]
fn test_chunk_empty_text_rejected() {
    // Empty chunk should be rejected or handled
}
```

### 1.4 Embedding (`crates/cp-core/src/embedding.rs`)

```rust
#[test]
fn test_embedding_new() {
    // Create embedding with valid vector
}

#[test]
fn test_embedding_id_derivation() {
    // embedding_id = generate_id(chunk_id || model_hash || version)
}

#[test]
fn test_embedding_i16_vector_format() {
    // Vector stored as i16 array
}

#[test]
fn test_embedding_model_hash_binding() {
    // model_hash binds to specific model
}

#[test]
fn test_embedding_version_increments() {
    // embedding_version tracks re-embedding
}

#[test]
fn test_embedding_l2_norm_precomputation() {
    // l2_norm stored for cosine similarity
}

#[test]
fn test_embedding_canonical_bytes() {
    // Verify serialization includes all fields
}

#[test]
fn test_embedding_dimension_validation() {
    // 384 dimensions for MiniLM-L6-v2
}

#[test]
fn test_embedding_quantization_bounds() {
    // i16 range: -32767 to 32767
}
```

### 1.5 Edge (`crates/cp-core/src/edge.rs`)

```rust
#[test]
fn test_edge_new_doc_to_chunk() {
    // Document -> Chunk relationship
}

#[test]
fn test_edge_new_chunk_to_embedding() {
    // Chunk -> Embedding relationship
}

#[test]
fn test_edge_new_chunk_to_chunk() {
    // Semantic similarity between chunks
}

#[test]
fn test_edge_with_weight() {
    // Optional weight field
}

#[test]
fn test_edge_custom_kind() {
    // Custom(String) edge kind
}

#[test]
fn test_edge_canonical_bytes() {
    // source_id || target_id || kind || weight_bytes
}

#[test]
fn test_edge_uniqueness_constraint() {
    // (source_id, target_id, kind) must be unique
}

#[test]
fn test_edge_kind_serialization() {
    // EdgeKind serialization
}

#[test]
fn test_edge_delete_cascades() {
    // Deleting source/target affects edges
}
```

### 1.6 State & Merkle Tree (`crates/cp-core/src/state.rs`)

```rust
#[test]
fn test_state_root_new() {
    // Create state root
}

#[test]
fn test_compute_merkle_root_empty() {
    // Empty entity list -> BLAKE3(b"empty")
}

#[test]
fn test_compute_merkle_root_single_entity() {
    // Single entity returns its hash
}

#[test]
fn test_compute_merkle_root_odd_number_entities() {
    // Odd leaf: duplicate last leaf (hash(leaf || leaf))
}

#[test]
fn test_compute_merkle_root_sorted_order() {
    // Entities must be sorted by ID
}

#[test]
fn test_compute_merkle_root_deterministic() {
    // Same entities always produce same root
}

#[test]
fn test_compute_state_root_composition() {
    // doc_root || chunk_root || emb_root || edge_root
}

#[test]
fn test_state_root_parent_chain() {
    // parent_hash links to previous state
}

#[test]
fn test_state_root_sequence_number() {
    // Monotonic sequence
}

#[test]
fn test_state_root_signature() {
    // Ed25519 signature verification
}

#[test]
fn test_merkle_proof_generation() {
    // Generate proof for specific entity
}

#[test]
fn test_merkle_proof_verification_valid() {
    // Verify valid proof
}

#[test]
fn test_merkle_proof_verification_invalid() {
    // Verify proof fails for tampered data
}

#[test]
fn test_merkle_proof_inclusion() {
    // Prove entity is in tree
}
```

### 1.7 HLC (Hybrid Logical Clock) (`crates/cp-core/src/hlc.rs`)

```rust
#[test]
fn test_hlc_new() {
    // Create new HLC timestamp
}

#[test]
fn test_hlc_now_increments() {
    // Multiple calls should increment
}

#[test]
fn test_hlc_send_receive_ordering() {
    // Receive updates clock if lower
}

#[test]
fn test_hlc_causality_preserved() {
    // HLC respects causality
}

#[test]
fn test_hlc_concurrent_events_tiebreaker() {
    // Same timestamp uses node_id tiebreaker
}

#[test]
fn test_hlc_logical_greater_than_physical() {
    // Logical component takes precedence
}

#[test]
fn test_hlc_serialization() {
    // Round-trip serialization
}

#[test]
fn test_hlc_parse_from_bytes() {
    // Deserialize HLC
}

#[test]
fn test_hlc_max_value() {
    // Boundary condition
}

#[test]
fn test_hlc_zero_value() {
    // Initial/zero timestamp
}
```

### 1.8 CognitiveDiff (`crates/cp-core/src/diff.rs`)

```rust
#[test]
fn test_cognitive_diff_empty() {
    // Create empty diff
}

#[test]
fn test_cognitive_diff_add_document() {
    // Add document to diff
}

#[test]
fn test_cognitive_diff_add_chunk() {
    // Add chunk to diff
}

#[test]
fn test_cognitive_diff_add_embedding() {
    // Add embedding to diff
}

#[test]
fn test_cognitive_diff_add_edge() {
    // Add edge to diff
}

#[test]
fn test_cognitive_diff_remove_document() {
    // Mark document for removal
}

#[test]
fn test_cognitive_diff_remove_chunk() {
    // Mark chunk for removal
}

#[test]
fn test_cognitive_diff_remove_embedding() {
    // Mark embedding for removal
}

#[test]
fn test_cognitive_diff_remove_edge() {
    // Mark edge for removal
}

#[test]
fn test_cognitive_diff_metadata() {
    // prev_root, new_root, timestamp, device_id, sequence
}

#[test]
fn test_cognitive_diff_serialization_cbor() {
    // CBOR encoding
}

#[test]
fn test_cognitive_diff_compression_zstd() {
    // zstd compression
}

#[test]
fn test_cognitive_diff_round_trip() {
    // Serialize -> deserialize produces identical diff
}

#[test]
fn test_cognitive_diff_apply_order() {
    // Verify correct application order
}

#[test]
fn test_cognitive_diff_idempotent_apply() {
    // Applying same diff twice has no additional effect
}
```

### 1.9 Text Canonicalization (`crates/cp-core/src/text.rs`)

```rust
#[test]
fn test_text_canonicalize_lowercase() {
    // Convert to lowercase
}

#[test]
fn test_text_canonicalize_whitespace() {
    // Normalize whitespace
}

#[test]
fn test_text_canonicalize_unicode() {
    // Unicode normalization
}

#[test]
fn test_text_canonicalize_empty() {
    // Empty string handling
}

#[test]
fn test_text_canonicalize_special_characters() {
    // Preserve meaningful special chars
}

#[test]
fn test_text_canonicalize_determinism() {
    // Same input always same output
}

#[test]
fn test_text_token_count_estimation() {
    // Estimate tokens for budgeting
}

#[test]
fn test_text_truncation() {
    // Max length truncation
}
```

### 1.10 Context Assembler (`crates/cp-core/src/context_assembler.rs`)

```rust
#[test]
fn test_context_assembler_new() {
    // Create with token budget
}

#[test]
fn test_context_assemble_deterministic_order() {
    // Same chunks, different input order -> same output
}

#[test]
fn test_context_assemble_respects_token_budget() {
    // Don't exceed budget
}

#[test]
fn test_context_assemble_highest_scored_first() {
    // Priority to higher scores
}

#[test]
fn test_context_assemble_metadata_query_hash() {
    // Query hash in metadata
}

#[test]
fn test_context_assemble_metadata_state_root() {
    // State root in metadata
}

#[test]
fn test_context_formatting_citations() {
    // Include document path citations
}

#[test]
fn test_context_formatting_chunk_separation() {
    // Chunks separated appropriately
}

#[test]
fn test_context_empty_chunks() {
    // Handle empty chunk list
}

#[test]
fn test_context_all_chunks_exceed_budget() {
    // When all chunks exceed budget, select highest scored
}

#[test]
fn test_context_token_budget_calculation() {
    // Verify token counting
}

#[test]
fn test_context_reserved_tokens_system() {
    // Reserved system tokens
}

#[test]
fn test_context_reserved_tokens_query() {
    // Reserved query tokens
}

#[test]
fn test_context_reserved_tokens_response() {
    // Reserved response tokens
}
```

---

## 2. cp-graph - Storage Engine

### 2.1 GraphStore (`crates/cp-graph/src/lib.rs`)

```rust
#[test]
fn test_graph_store_new() {
    // Create new GraphStore
}

#[test]
fn test_graph_store_in_memory() {
    // In-memory store for testing
}

#[test]
fn test_graph_store_insert_document() {
    // Insert document
}

#[test]
fn test_graph_store_get_document_by_id() {
    // Retrieve document by ID
}

#[test]
fn test_graph_store_get_document_by_path() {
    // Retrieve document by path
}

#[test]
fn test_graph_store_document_not_found() {
    // Handle missing document
}

#[test]
fn test_graph_store_update_document() {
    // Update existing document
}

#[test]
fn test_graph_store_delete_document() {
    // Delete document cascades to chunks/embeddings
}

#[test]
fn test_graph_store_all_documents() {
    // List all documents
}

#[test]
fn test_graph_store_insert_chunk() {
    // Insert chunk
}

#[test]
fn test_graph_store_get_chunk() {
    // Get chunk by ID
}

#[test]
fn test_graph_store_chunks_for_document() {
    // Get all chunks for document
}

#[test]
fn test_graph_store_delete_chunks_for_document() {
    // Cascade delete chunks
}

#[test]
fn test_graph_store_insert_embedding() {
    // Insert embedding
}

#[test]
fn test_graph_store_get_embedding() {
    // Get embedding by ID
}

#[test]
fn test_graph_store_embeddings_for_chunk() {
    // Get embeddings for chunk
}

#[test]
fn test_graph_store_all_embeddings() {
    // List all embeddings
}

#[test]
fn test_graph_store_insert_edge() {
    // Insert edge
}

#[test]
fn test_graph_store_edges_from() {
    // Get edges from source
}

#[test]
fn test_graph_store_edges_to() {
    // Get edges to target
}

#[test]
fn test_graph_store_all_edges() {
    // List all edges
}

#[test]
fn test_graph_store_compute_merkle_root() {
    // Compute state root
}

#[test]
fn test_graph_store_insert_state_root() {
    // Insert state root
}

#[test]
fn test_graph_store_get_latest_state_root() {
    // Get most recent state root
}

#[test]
fn test_graph_store_get_state_root_by_hash() {
    // Get by hash
}

#[test]
fn test_graph_store_stats() {
    // Get graph statistics
}

#[test]
fn test_graph_store_stats_empty() {
    // Stats for empty graph
}
```

### 2.2 Transactions (`crates/cp-graph/src/lib.rs`)

```rust
#[test]
fn test_transaction_begin() {
    // Begin transaction
}

#[test]
fn test_transaction_commit() {
    // Commit changes
}

#[test]
fn test_transaction_rollback() {
    // Rollback on error
}

#[test]
fn test_transaction_atomicity() {
    // All or nothing
}

#[test]
fn test_transaction_isolation() {
    // Concurrent reads see consistent state
}

#[test]
fn test_transaction_nested_rejected() {
    // Nested transactions not supported
}

#[test]
fn test_transaction_auto_rollback_on_panic() {
    // Panic triggers rollback
}
```

### 2.3 Migrations (`crates/cp-graph/src/migrations.rs`)

```rust
#[test]
fn test_migration_runner_initial_schema() {
    // Apply initial schema
}

#[test]
fn test_migration_already_applied() {
    // Skip already applied migrations
}

#[test]
fn test_migration_missing_version() {
    // Handle missing version
}

#[test]
fn test_migration_rollback_on_failure() {
    // Rollback if migration fails
}

#[test]
fn test_migration_001_documents_table() {
    // Verify documents table schema
}

#[test]
fn test_migration_002_timestamps() {
    // Verify timestamps added
}

#[test]
fn test_migration_003_l2_norm() {
    // Verify l2_norm column
}

#[test]
fn test_migration_004_path_id_embedding_version() {
    // Verify path_id and embedding_version
}

#[test]
fn test_migration_version_tracking() {
    // schema_version table updated
}

#[test]
fn test_migration_foreign_keys() {
    // CASCADE delete constraints
}

#[test]
fn test_migration_fts_triggers() {
    // FTS5 triggers configured
}
```

### 2.4 HNSW Index (`crates/cp-graph/src/index.rs`)

```rust
#[test]
fn test_hnsw_index_new() {
    // Create new index
}

#[test]
fn test_hnsw_index_add_vector() {
    // Add vector to index
}

#[test]
fn test_hnsw_index_search() {
    // Search for similar vectors
}

#[test]
fn test_hnsw_index_search_k_results() {
    // Return top K results
}

#[test]
fn test_hnsw_index_search_empty_query() {
    // Handle empty query
}

#[test]
fn test_hnsw_index_delete_vector() {
    // Mark vector as deleted
}

#[test]
fn test_hnsw_index_persistence_save() {
    // Save index to disk
}

#[test]
fn test_hnsw_index_persistence_load() {
    // Load index from disk
}

#[test]
fn test_hnsw_index_persistence_corrupted() {
    // Handle corrupted index file
}

#[test]
fn test_hnsw_index_rebuild_from_sqlite() {
    // Rebuild from SQLite source
}

#[test]
fn test_hnsw_index_incremental_add() {
    // Add to existing index
}

#[test]
fn test_hnsw_index_consistency_with_sqlite() {
    // Count matches SQLite
}

#[test]
fn test_hnsw_index_cosine_similarity() {
    // Verify cosine distance
}

#[test]
fn test_hnsw_index_empty_index_search() {
    // Search on empty index
}

#[test]
fn test_hnsw_index_batch_add() {
    // Add multiple vectors
}

#[test]
fn test_hnsw_index_m_configuration() {
    // M = 16 connections
}

#[test]
fn test_hnsw_index_ef_configuration() {
    // ef_construction = 200
}
```

### 2.5 FTS (Full-Text Search)

```rust
#[test]
fn test_fts_search_basic() {
    // Basic keyword search
}

#[test]
fn test_fts_search_multiple_terms() {
    // Multiple keyword AND/OR
}

#[test]
fn test_fts_search_no_results() {
    // No matching documents
}

#[test]
fn test_fts_search_ranking() {
    // BM25 ranking
}

#[test]
fn test_fts_search_porter_stemming() {
    // Stemming works
}

#[test]
fn test_fts_search_unicode() {
    // Unicode tokenization
}

#[test]
fn test_fts_trigger_insert() {
    // Insert triggers FTS update
}

#[test]
fn test_fts_trigger_delete() {
    // Delete triggers FTS update
}

#[test]
fn test_fts_trigger_update() {
    // Update triggers FTS update
}
```

---

## 3. cp-canonical-embeddings - Embedding Generation

### 3.1 Tokenizer (`crates/cp-canonical-embeddings/src/tokenizer_impl.rs`)

```rust
#[test]
fn test_tokenizer_encode_basic() {
    // Basic text tokenization
}

#[test]
fn test_tokenizer_encode_special_tokens() {
    // [CLS], [SEP], [PAD], [UNK] tokens
}

#[test]
fn test_tokenizer_encode_test_vector() {
    // CP-010 test vector: "Hello, world!" -> [101, 7592, 1010, 2088, 999, 102]
}

#[test]
fn test_tokenizer_encode_lowercase() {
    // Uncased model converts to lowercase
}

#[test]
fn test_tokenizer_encode_truncation() {
    // Truncate to max_seq_len (256)
}

#[test]
fn test_tokenizer_encode_empty() {
    // Empty string
}

#[test]
fn test_tokenizer_decode() {
    // Decode token IDs back to text
}

#[test]
fn test_tokenizer_vocab_size() {
    // 30522 for MiniLM-L6-v2
}

#[test]
fn test_tokenizer_unknown_token() {
    // Unknown tokens -> [UNK]
}

#[test]
fn test_tokenizer_padding() {
    // Pad to batch max length
}

#[test]
fn test_tokenizer_attention_mask() {
    // Create attention mask
}

#[test]
fn test_tokenizer_batch_encode() {
    // Batch encoding
}

#[test]
fn test_tokenizer_determinism() {
    // Same text -> same tokens
}
```

### 3.2 Model (`crates/cp-canonical-embeddings/src/model.rs`)

```rust
#[test]
fn test_model_load() {
    // Load MiniLM-L6-v2 model
}

#[test]
fn test_model_forward_pass() {
    // Forward pass produces hidden states
}

#[test]
fn test_model_output_dimension() {
    // 384 dimensions
}

#[test]
fn test_model_max_sequence_length() {
    // 256 tokens
}

#[test]
fn test_model_mean_pooling() {
    // Mean pooling implementation
}

#[test]
fn test_model_l2_normalization() {
    // L2 normalize output
}

#[test]
fn test_model_quantization_f32_to_i16() {
    // Quantize to i16
}

#[test]
fn test_model_quantization_bounds() {
    // -32767 to 32767
}

#[test]
fn test_model_quantization_rounding() {
    // round_half_to_even
}

#[test]
fn test_model_canonical_embedding_generation() {
    // Full pipeline
}

#[test]
fn test_model_batch_embedding() {
    // Batch processing
}

#[test]
fn test_model_hash() {
    // Verify model_hash = BLAKE3(model_id)
}

#[test]
fn test_model_dropout_disabled() {
    // No dropout in inference
}

#[test]
fn test_model_determinism() {
    // Same input -> same output
}

#[test]
fn test_model_determinism_across_runs() {
    // Run multiple times, same result
}

#[test]
fn test_model_embedding_test_vector() {
    // CP-010 test vector: "The quick brown fox"
}
```

### 3.3 SoftFloat (`crates/cp-canonical-embeddings/src/softfloat.rs`)

```rust
#[test]
fn test_softfloat_add() {
    // SoftFloat addition
}

#[test]
fn test_softfloat_subtract() {
    // SoftFloat subtraction
}

#[test]
fn test_softfloat_multiply() {
    // SoftFloat multiplication
}

#[test]
fn test_softfloat_divide() {
    // SoftFloat division
}

#[test]
fn test_softfloat_sqrt() {
    // SoftFloat square root
}

#[test]
fn test_softfloat_compare() {
    // Comparison operations
}

#[test]
fn test_softfloat_from_f32() {
    // Convert from f32
}

#[test]
fn test_softfloat_to_f32() {
    // Convert to f32
}

#[test]
fn test_softfloat_determinism() {
    // Same input -> same output across platforms
}

#[test]
fn test_softfloat_special_values() {
    // NaN, Infinity handling
}

#[test]
fn test_softfloat_zero() {
    // Zero handling
}

#[test]
fn test_softfloat_l2_norm() {
    // Using SoftFloat for L2 computation
}

#[test]
fn test_softfloat_division_by_zero() {
    // Handle division by zero
}
```

---

## 4. cp-parser - Document Parsing

### 4.1 Parsers (`crates/cp-parser/src/parsers.rs`)

```rust
#[test]
fn test_parse_markdown() {
    // Parse markdown document
}

#[test]
fn test_parse_plain_text() {
    // Parse plain text
}

#[test]
fn test_parse_markdown_headers() {
    // Headers extracted
}

#[test]
fn test_parse_markdown_code_blocks() {
    // Code blocks handled
}

#[test]
fn test_parse_markdown_lists() {
    // List items handled
}

#[test]
fn test_parse_markdown_links() {
    // Links extracted
}

#[test]
fn test_parse_invalid_utf8() {
    // Invalid UTF-8 rejection
}

#[test]
fn test_parse_empty_document() {
    // Empty file
}

#[test]
fn test_parse_large_document() {
    // Large file handling
}

#[test]
fn test_parse_binary_file() {
    // Binary file rejection
}

#[test]
fn test_parse_mime_type_detection() {
    // MIME type detection
}
```

### 4.2 Chunker (`crates/cp-parser/src/chunker.rs`)

```rust
#[test]
fn test_chunker_by_tokens() {
    // Chunk by token count
}

#[test]
fn test_chunker_by_sentence() {
    // Chunk by sentence boundaries
}

#[test]
fn test_chunker_by_paragraph() {
    // Chunk by paragraphs
}

#[test]
fn test_chunker_overlap() {
    // Overlapping chunks
}

#[test]
fn test_chunker_max_chunk_size() {
    // Maximum chunk size enforcement
}

#[test]
fn test_chunker_min_chunk_size() {
    // Minimum chunk size
}

#[test]
fn test_chunker_preserves_boundaries() {
    // Don't split mid-sentence (when possible)
}

#[test]
fn test_chunker_empty_content() {
    // Empty content
}

#[test]
fn test_chunker_single_token() {
    // Single token content
}

#[test]
fn test_chunker_unicode_content() {
    // Unicode text chunking
}

#[test]
fn test_chunker_preserves_context() {
    // Context around boundaries
}
```

---

## 5. cp-query - Query Engine

### 5.1 QueryEngine (`crates/cp-query/src/lib.rs`)

```rust
#[test]
fn test_query_engine_new() {
    // Create query engine
}

#[test]
fn test_query_engine_search_semantic() {
    // Vector search only
}

#[test]
fn test_query_engine_search_lexical() {
    // FTS search only
}

#[test]
fn test_query_engine_search_hybrid() {
    // Combined semantic + lexical
}

#[test]
fn test_query_engine_rrf_fusion() {
    // Reciprocal Rank Fusion
}

#[test]
fn test_query_engine_weight_semantic() {
    // Semantic weight
}

#[test]
fn test_query_engine_weight_lexical() {
    // Lexical weight
}

#[test]
fn test_query_engine_filter_by_path() {
    // Filter by document path
}

#[test]
fn test_query_engine_filter_by_date() {
    // Filter by date range
}

#[test]
fn test_query_engine_limit_results() {
    // Limit number of results
}

#[test]
fn test_query_engine_no_results() {
    // Empty results
}

#[test]
fn test_query_engine_cache_hit() {
    // Cache hit
}

#[test]
fn test_query_engine_cache_miss() {
    // Cache miss -> compute
}

#[test]
fn test_query_engine_cache_invalidation() {
    // Invalidate on new content
}

#[test]
fn test_query_engine_score_calculation() {
    // Score computation
}

#[test]
fn test_query_engine_result_ranking() {
    // Results ranked by score
}

#[test]
fn test_query_engine_empty_query() {
    // Empty query handling
}

#[test]
fn test_query_engine_special_characters() {
    // Special chars in query
}
```

---

## 6. cp-sync - Synchronization

### 6.1 Crypto (`crates/cp-sync/src/crypto.rs`)

```rust
#[test]
fn test_crypto_generate_keypair() {
    // Ed25519 keypair generation
}

#[test]
fn test_crypto_sign() {
    // Sign message
}

#[test]
fn test_crypto_verify_valid() {
    // Verify valid signature
}

#[test]
fn test_crypto_verify_invalid() {
    // Verify invalid signature
}

#[test]
fn test_crypto_encrypt_xchacha20() {
    // XChaCha20-Poly1305 encryption
}

#[test]
fn test_crypto_decrypt_xchacha20() {
    // Decrypt
}

#[test]
fn test_crypto_encrypt_decrypt_roundtrip() {
    // Round-trip
}

#[test]
fn test_crypto_nonce_uniqueness() {
    // Unique nonces required
}

#[test]
fn test_crypto_key_derivation_hkdf() {
    // HKDF key derivation
}

#[test]
fn test_crypto_test_vector_encryption() {
    // CP-013 test vector
}

#[test]
fn test_crypto_test_vector_signature() {
    // CP-013 test vector
}

#[test]
fn test_crypto_tampering_detection() {
    // Tampered ciphertext detected
}

#[test]
fn test_crypto_wrong_key_rejected() {
    // Wrong key fails
}
```

### 6.2 Merkle (`crates/cp-sync/src/merkle.rs`)

```rust
#[test]
fn test_merkle_diff_generate() {
    // Generate diff between states
}

#[test]
fn test_merkle_diff_empty() {
    // No changes
}

#[test]
fn test_merkle_diff_document_added() {
    // Document added
}

#[test]
fn test_merkle_diff_document_modified() {
    // Document modified
}

#[test]
fn test_merkle_diff_document_deleted() {
    // Document deleted
}

#[test]
fn test_merkle_diff_chunk_changes() {
    // Chunk changes
}

#[test]
fn test_merkle_diff_embedding_changes() {
    // Embedding changes
}

#[test]
fn test_merkle_diff_edge_changes() {
    // Edge changes
}

#[test]
fn test_merkle_diff_multiple_changes() {
    // Multiple entity types changed
}

#[test]
fn test_merkle_diff_serialization() {
    // Serialize diff
}

#[test]
fn test_merkle_diff_apply() {
    // Apply diff to graph
}

#[test]
fn test_merkle_diff_apply_order() {
    // Correct application order
}

#[test]
fn test_merkle_diff_idempotent() {
    // Apply same diff twice
}
```

### 6.3 Identity (`crates/cp-sync/src/identity.rs`)

```rust
#[test]
fn test_identity_generate() {
    // Generate device identity
}

#[test]
fn test_identity_public_key_derivation() {
    // Public key from private key
}

#[test]
fn test_identity_device_id_derivation() {
    // device_id = generate_id(public_key)
}

#[test]
fn test_identity_serialization() {
    // Serialize identity
}

#[test]
fn test_identity_persistence() {
    // Save/load identity
}

#[test]
fn test_identity_pairing_x25519() {
    // X25519 key agreement
}

#[test]
fn test_identity_shared_key_derivation() {
    // Derive shared encryption key
}

#[test]
fn test_identity_unpairing() {
    // Remove paired device
}
```

---

## 7. cp-relay-server - Relay Server

### 7.1 Storage (`relay/cp-relay-server/src/storage.rs`)

```rust
#[test]
fn test_relay_storage_push() {
    // Push diff to storage
}

#[test]
fn test_relay_storage_pull() {
    // Pull diffs since sequence
}

#[test]
fn test_relay_storage_acknowledge() {
    // Acknowledge received diffs
}

#[test]
fn test_relay_storage_sequence_tracking() {
    // Track sequences
}

#[test]
fn test_relay_storage_retention() {
    // Delete old diffs after 30 days
}

#[test]
fn test_relay_storage_not_found() {
    // Missing diffs
}
```

### 7.2 Server (`relay/cp-relay-server/src/lib.rs`)

```rust
#[test]
fn test_relay_push_endpoint() {
    // POST /push
}

#[test]
fn test_relay_pull_endpoint() {
    // GET /pull
}

#[test]
fn test_relay_delete_endpoint() {
    // DELETE /acknowledge
}

#[test]
fn test_relay_authentication() {
    // X-Device-ID header validation
}

#[test]
fn test_relay_rate_limiting() {
    // Rate limit enforcement
}

#[test]
fn test_relay_quota_enforcement() {
    // Storage quota
}

#[test]
fn test_relay_concurrent_push() {
    // Multiple device pushes
}

#[test]
fn test_relay_health_check() {
    // Health endpoint
}
```

---

## 8. cp-desktop - Desktop Application

### 8.1 File Watcher (`crates/cp-desktop/src/watcher.rs`)

```rust
#[test]
fn test_watcher_new() {
    // Create watcher
}

#[test]
fn test_watcher_add_directory() {
    // Watch directory
}

#[test]
fn test_watcher_file_created() {
    // Detect new file
}

#[test]
fn test_watcher_file_modified() {
    // Detect file modification
}

#[test]
fn test_watcher_file_deleted() {
    // Detect file deletion
}

#[test]
fn test_watcher_file_moved() {
    // Detect file move/rename
}

#[test]
fn test_watcher_ignore_patterns() {
    // Ignore .git, node_modules, etc.
}

#[test]
fn test_watcher_debounce() {
    // Debounce rapid changes
}

#[test]
fn test_watcher_stop() {
    // Stop watching
}
```

### 8.2 DesktopApp (`crates/cp-desktop/src/lib.rs`)

```rust
#[test]
fn test_desktop_app_new() {
    // Create desktop app
}

#[test]
fn test_desktop_app_add_watch_dir() {
    // Add directory to watch
}

#[test]
fn test_desktop_app_start() {
    // Start app
}

#[test]
fn test_desktop_app_stop() {
    // Stop app
}

#[test]
fn test_desktop_app_graph_access() {
    // Access graph store
}

#[test]
fn test_desktop_app_ingest_file() {
    // Manual file ingest
}

#[test]
fn test_desktop_app_remove_file() {
    // Remove file from watch
}
```

---

## 9. Error Handling Tests (All Crates)

```rust
#[test]
fn test_error_invalid_input() {
    // Invalid input rejection
}

#[test]
fn test_error_database_connection_failed() {
    // DB connection failure
}

#[test]
fn test_error_serialization_failed() {
    // Serialization errors
}

#[test]
fn test_error_deserialization_failed() {
    // Deserialization errors
}

#[test]
fn test_error_not_found() {
    // Entity not found
}

#[test]
fn test_error_already_exists() {
    // Duplicate entity
}

#[test]
fn test_error_constraint_violation() {
    // Constraint violation
}

#[test]
fn test_error_out_of_memory() {
    // Memory pressure handling
}

#[test]
fn test_error_disk_full() {
    // Disk full handling
}

#[test]
fn test_error_permission_denied() {
    // Permission errors
}

#[test]
fn test_error_timeout() {
    // Operation timeout
}

#[test]
fn test_error_cancellation() {
    // Operation cancelled
}
```

---

## 10. Integration Tests (`crates/cp-tests/tests/`)

```rust
#[tokio::test]
async fn test_integration_full_ingestion_workflow() {
    // File -> Parse -> Chunk -> Embed -> Store -> Index
}

#[tokio::test]
async fn test_integration_deterministic_rebuild() {
    // Full rebuild produces same state root
}

#[tokio::test]
async fn test_integration_incremental_update() {
    // Add file produces minimal diff
}

#[tokio::test]
async fn test_integration_delete_cascade() {
    // Delete document cascades properly
}

#[tokio::test]
async fn test_integration_sync_two_devices() {
    // Two devices converge to same state
}

#[tokio::test]
async fn test_integration_sync_conflict_resolution() {
    // Concurrent edits resolved
}

#[tokio::test]
async fn test_integration_search_roundtrip() {
    // Query -> Search -> Retrieve -> Display
}

#[tokio::test]
async fn test_integration_context_assembly_workflow() {
    // Query -> Retrieve -> Assemble -> Format
}

#[tokio::test]
async fn test_integration_hybrid_search() {
    // Vector + Lexical combined
}

#[tokio::test]
async fn test_integration_backup_restore() {
    // Export -> Import produces identical state
}

#[tokio::test]
async fn test_integration_migration_upgrade() {
    // Old schema -> New schema
}

#[tokio::test]
async fn test_integration_large_dataset() {
    // 10k+ documents performance
}

#[tokio::test]
async fn test_integration_concurrent_writes() {
    // Multiple writers
}

#[tokio::test]
async fn test_integration_offline_then_sync() {
    // Offline edits sync when online
}
```

---

## Summary

| Category | Test Scenarios |
|----------|---------------|
| cp-core (ID, Doc, Chunk, Embedding, Edge, State, HLC, Diff, Text, Context) | 150+ |
| cp-graph (Store, Transactions, Migrations, Index, FTS) | 80+ |
| cp-canonical-embeddings (Tokenizer, Model, SoftFloat) | 50+ |
| cp-parser (Parsers, Chunker) | 25+ |
| cp-query (QueryEngine) | 25+ |
| cp-sync (Crypto, Merkle, Identity) | 40+ |
| cp-relay-server (Storage, Endpoints) | 20+ |
| cp-desktop (Watcher, App) | 15+ |
| Error Handling | 20+ |
| Integration Tests | 15+ |

**Total: ~445 test scenarios** for comprehensive 100% coverage.

---

## Test Organization Recommendations

### File Structure
```
crates/
├── cp-core/
│   ├── src/
│   │   ├── id.rs
│   │   ├── document.rs
│   │   └── ...
│   └── tests/
│       ├── id_tests.rs
│       ├── document_tests.rs
│       ├── chunk_tests.rs
│       ├── embedding_tests.rs
│       ├── edge_tests.rs
│       ├── state_tests.rs
│       ├── hlc_tests.rs
│       ├── diff_tests.rs
│       ├── text_tests.rs
│       └── context_tests.rs
├── cp-graph/
│   └── tests/
│       ├── graph_store_tests.rs
│       ├── transaction_tests.rs
│       ├── migration_tests.rs
│       ├── hnsw_index_tests.rs
│       └── fts_tests.rs
├── cp-canonical-embeddings/
│   └── tests/
│       ├── tokenizer_tests.rs
│       ├── model_tests.rs
│       └── softfloat_tests.rs
├── cp-parser/
│   └── tests/
│       ├── parser_tests.rs
│       └── chunker_tests.rs
├── cp-query/
│   └── tests/
│       └── query_engine_tests.rs
├── cp-sync/
│   └── tests/
│       ├── crypto_tests.rs
│       ├── merkle_tests.rs
│       └── identity_tests.rs
└── cp-tests/
    └── tests/
        ├── full_ingestion_tests.rs
        ├── deterministic_rebuild_tests.rs
        ├── sync_tests.rs
        └── performance_tests.rs
```

### Test Naming Conventions
- Unit tests: `#[test] fn test_<component>_<behavior>()`
- Integration tests: `#[tokio::test] async fn test_integration_<workflow>()`
- Property tests: `#[proptest] fn prop_<component>_<property>()`

### Test Fixtures
- Use `tempfile::TempDir` for file-based tests
- Use `once_cell::sync::Lazy` for static fixtures
- Create deterministic test data generators
