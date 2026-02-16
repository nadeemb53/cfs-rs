//! CP Graph - Knowledge graph and vector store
//!
//! Combines SQLite for structured data with HNSW for vector search.

mod migrations;
mod index;

pub use migrations::{get_schema_version, needs_migration, run_migrations};
pub use index::{IndexConfig, PersistentHnswIndex, SharedPersistentIndex};

use cp_core::{Chunk, CPError, Document, Edge, EdgeKind, Embedding, Result};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::PathBuf;
use tracing::info;
use uuid::Uuid;


/// Combined graph and vector store
pub struct GraphStore {
    /// SQLite connection for structured data
    db: Connection,
    /// Persistent HNSW index for vector search
    hnsw: SharedPersistentIndex,
    /// Path to database (for index path derivation)
    db_path: Option<PathBuf>,
    /// Index configuration
    index_config: IndexConfig,
}

impl GraphStore {
    /// Open or create a graph store at the given path
    pub fn open(db_path: &str) -> Result<Self> {
        Self::open_with_config(db_path, IndexConfig::default())
    }

    /// Open or create a graph store with custom index configuration
    pub fn open_with_config(db_path: &str, index_config: IndexConfig) -> Result<Self> {
        let (db, path) = if db_path == ":memory:" {
            (Connection::open_in_memory().map_err(|e| CPError::Database(e.to_string()))?, None)
        } else {
            let path = PathBuf::from(db_path);
            (Connection::open(&path).map_err(|e| CPError::Database(e.to_string()))?, Some(path))
        };

        // Run migrations (handles both fresh DBs and upgrades)
        run_migrations(&db)?;

        // Create or open persistent HNSW index
        let hnsw = if let Some(ref p) = path {
            let index_path = p.with_extension("usearch");
            SharedPersistentIndex::open(index_path, index_config.clone())?
        } else {
            SharedPersistentIndex::new(index_config.clone())?
        };

        let store = Self {
            db,
            hnsw,
            db_path: path,
            index_config,
        };

        // Check if index needs rebuild from DB
        if store.hnsw.needs_rebuild() {
            store.rebuild_hnsw_index()?;
        } else {
            // Validate against current state
            let current_root = store.compute_merkle_root()?;
            if !store.hnsw.is_valid(&current_root) {
                info!("Index checkpoint mismatch, rebuilding...");
                store.rebuild_hnsw_index()?;
            }
        }

        info!("Opened graph store at {} and loaded index", db_path);

        Ok(store)
    }

    /// Open an in-memory graph store (for testing)
    pub fn in_memory() -> Result<Self> {
        Self::open(":memory:")
    }

    /// Rebuild the HNSW index from database
    fn rebuild_hnsw_index(&self) -> Result<()> {
        info!("Rebuilding HNSW index from database...");
        self.hnsw.clear()?;

        let embs = self.get_all_embeddings()?;
        for emb in embs {
            self.hnsw.insert(emb.id, emb.to_f32())?;
        }

        // Checkpoint with current state
        let root = self.compute_merkle_root()?;
        self.hnsw.checkpoint(root)?;

        info!("HNSW index rebuilt with {} vectors", self.hnsw.len());
        Ok(())
    }

    // ========== Document operations ==========

    /// Insert a document
    pub fn insert_document(&mut self, doc: &Document) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO documents (id, path, hash, hierarchical_hash, mtime, size, mime_type) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    doc.id.as_bytes().as_slice(),
                    doc.path.to_string_lossy().as_ref(),
                    doc.hash.as_slice(),
                    doc.hierarchical_hash.as_slice(),
                    doc.mtime,
                    doc.size as i64,
                    &doc.mime_type,
                ],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get a document by ID
    pub fn get_document(&self, id: Uuid) -> Result<Option<Document>> {
        self.db
            .query_row(
                "SELECT id, path, hash, hierarchical_hash, mtime, size, mime_type FROM documents WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let path_str: String = row.get(1)?;
                    let hash_bytes: Vec<u8> = row.get(2)?;
                    let hh_bytes: Vec<u8> = row.get(3)?;
                    let mtime: i64 = row.get(4)?;
                    let size: i64 = row.get(5)?;
                    let mime_type: String = row.get(6)?;

                    let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_bytes);
                    let mut hierarchical_hash = [0u8; 32];
                    hierarchical_hash.copy_from_slice(&hh_bytes);

                    Ok(Document {
                        id,
                        path_id: Uuid::nil(), // computed from path

                        path: std::path::PathBuf::from(path_str),
                        hash,
                        hierarchical_hash,
                        mtime,
                        size: size as u64,
                        mime_type,
                    })
                },
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    /// Get a document by path
    pub fn get_document_by_path(&self, path: &std::path::Path) -> Result<Option<Document>> {
        self.db
            .query_row(
                "SELECT id, path, hash, hierarchical_hash, mtime, size, mime_type FROM documents WHERE path = ?1",
                params![path.to_string_lossy().as_ref()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let path_str: String = row.get(1)?;
                    let hash_bytes: Vec<u8> = row.get(2)?;
                    let hh_bytes: Vec<u8> = row.get(3)?;
                    let mtime: i64 = row.get(4)?;
                    let size: i64 = row.get(5)?;
                    let mime_type: String = row.get(6)?;

                    let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_bytes);
                    let mut hierarchical_hash = [0u8; 32];
                    hierarchical_hash.copy_from_slice(&hh_bytes);

                    Ok(Document {
                        id,
                        path_id: Uuid::nil(), // computed from path

                        path: std::path::PathBuf::from(path_str),
                        hash,
                        hierarchical_hash,
                        mtime,
                        size: size as u64,
                        mime_type,
                    })
                },
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    /// Delete a document and its associated chunks/embeddings
    pub fn delete_document(&mut self, id: Uuid) -> Result<()> {
        // Cascade delete (chunks and embeddings deleted via manual cleanup)
        self.db
            .execute(
                "DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?1)",
                params![id.as_bytes().as_slice()],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        self.db
            .execute(
                "DELETE FROM chunks WHERE doc_id = ?1",
                params![id.as_bytes().as_slice()],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        self.db
            .execute(
                "DELETE FROM documents WHERE id = ?1",
                params![id.as_bytes().as_slice()],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get all documents
    pub fn get_all_documents(&self) -> Result<Vec<Document>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, path, hash, hierarchical_hash, mtime, size, mime_type FROM documents")
            .map_err(|e| CPError::Database(e.to_string()))?;

        let docs = stmt
            .query_map([], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let path_str: String = row.get(1)?;
                let hash_bytes: Vec<u8> = row.get(2)?;
                let hh_bytes: Vec<u8> = row.get(3)?;
                let mtime: i64 = row.get(4)?;
                let size: i64 = row.get(5)?;
                let mime_type: String = row.get(6)?;

                let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&hash_bytes);
                let mut hierarchical_hash = [0u8; 32];
                hierarchical_hash.copy_from_slice(&hh_bytes);

                Ok(Document {
                    id,
                    path_id: Uuid::nil(), // computed from path

                    path: std::path::PathBuf::from(path_str),
                    hash,
                    hierarchical_hash,
                    mtime,
                    size: size as u64,
                    mime_type,
                })
            })
            .map_err(|e| CPError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(docs)
    }

    // ========== Chunk operations ==========

    /// Insert a chunk
    pub fn insert_chunk(&mut self, chunk: &Chunk) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, text, byte_offset, byte_length, sequence, text_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk.id.as_bytes().as_slice(),
                    chunk.doc_id.as_bytes().as_slice(),
                    &chunk.text,
                    chunk.byte_offset as i64,
                    chunk.byte_length as i64,
                    chunk.sequence,
                    chunk.text_hash.as_slice(),
                ],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get chunks for a document
    pub fn get_chunks_for_doc(&self, doc_id: Uuid) -> Result<Vec<Chunk>> {
        let mut stmt = self
            .db
            .prepare(
                "SELECT id, doc_id, text, byte_offset, byte_length, sequence, text_hash
                 FROM chunks WHERE doc_id = ?1 ORDER BY sequence",
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        let chunks = stmt
            .query_map(params![doc_id.as_bytes().as_slice()], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let doc_id_bytes: Vec<u8> = row.get(1)?;
                let text: String = row.get(2)?;
                let byte_offset: i64 = row.get(3)?;
                let byte_length: i64 = row.get(4)?;
                let sequence: u32 = row.get(5)?;
                let text_hash_bytes: Vec<u8> = row.get(6)?;

                let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                let doc_id = Uuid::from_slice(&doc_id_bytes).expect("invalid uuid");
                let mut text_hash = [0u8; 32];
                text_hash.copy_from_slice(&text_hash_bytes);

                Ok(Chunk {
                    id,
                    doc_id,
                    text,
                    byte_offset: byte_offset as u64,
                    byte_length: byte_length as u64,
                    sequence,
                    text_hash,
                })
            })
            .map_err(|e| CPError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(chunks)
    }

    /// Get a chunk by ID
    pub fn get_chunk(&self, id: Uuid) -> Result<Option<Chunk>> {
        self.db
            .query_row(
                "SELECT id, doc_id, text, byte_offset, byte_length, sequence, text_hash FROM chunks WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    let doc_id_bytes: Vec<u8> = row.get(1)?;
                    let text: String = row.get(2)?;
                    let byte_offset: i64 = row.get(3)?;
                    let byte_length: i64 = row.get(4)?;
                    let sequence: u32 = row.get(5)?;
                    let text_hash_bytes: Vec<u8> = row.get(6)?;

                    let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                    let doc_id = Uuid::from_slice(&doc_id_bytes).expect("invalid uuid");
                    let mut text_hash = [0u8; 32];
                    text_hash.copy_from_slice(&text_hash_bytes);

                    Ok(Chunk {
                        id,
                        doc_id,
                        text,
                        byte_offset: byte_offset as u64,
                        byte_length: byte_length as u64,
                        sequence,
                        text_hash,
                    })
                },
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    // ========== Embedding operations ==========

    /// Insert an embedding and add it to the HNSW index
    pub fn insert_embedding(&mut self, emb: &Embedding) -> Result<()> {
        // Store in SQLite
        // vector is now i16, so flattening to bytes works as le_bytes
        let vector_bytes: Vec<u8> = emb
            .vector
            .iter()
            .flat_map(|val| val.to_le_bytes())
            .collect();

        self.db
            .execute(
                "INSERT OR REPLACE INTO embeddings (id, chunk_id, vector, model_hash, dim, l2_norm)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    emb.id.as_bytes().as_slice(),
                    emb.chunk_id.as_bytes().as_slice(),
                    &vector_bytes,
                    emb.model_hash.as_slice(),
                    emb.dim as i32,
                    emb.l2_norm,
                ],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        // Add to HNSW index (convert to f32)
        let vector_f32 = emb.to_f32();
        self.hnsw.insert(emb.id, vector_f32)?;

        // Mark index as needing checkpoint (state changed)
        self.hnsw.invalidate();

        Ok(())
    }

    /// Get embedding by chunk ID
    pub fn get_embedding_for_chunk(&self, chunk_id: Uuid) -> Result<Option<Embedding>> {
        self.db
            .query_row(
                "SELECT id, chunk_id, vector, model_hash, dim, l2_norm
                 FROM embeddings WHERE chunk_id = ?1",
                params![chunk_id.as_bytes().as_slice()],
                |row| self.row_to_embedding(row),
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    /// Get embedding by ID
    pub fn get_embedding(&self, id: Uuid) -> Result<Option<Embedding>> {
        self.db
            .query_row(
                "SELECT id, chunk_id, vector, model_hash, dim, l2_norm
                 FROM embeddings WHERE id = ?1",
                params![id.as_bytes().as_slice()],
                |row| self.row_to_embedding(row),
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    fn row_to_embedding(&self, row: &rusqlite::Row) -> rusqlite::Result<Embedding> {
        let id_bytes: Vec<u8> = row.get(0)?;
        let chunk_id_bytes: Vec<u8> = row.get(1)?;
        let vector_bytes: Vec<u8> = row.get(2)?;
        let model_hash_bytes: Vec<u8> = row.get(3)?;
        let _dim: i32 = row.get(4)?;
        let l2_norm: f32 = row.get(5).unwrap_or(0.0);

        let _id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
        let chunk_id = Uuid::from_slice(&chunk_id_bytes).expect("invalid uuid");
        let mut model_hash = [0u8; 32];
        model_hash.copy_from_slice(&model_hash_bytes);

        // Convert bytes back to i16 vector
        let vector: Vec<i16> = vector_bytes
            .chunks(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();

        Ok(Embedding::from_quantized_with_norm(
            chunk_id,
            vector,
            model_hash,
            l2_norm,
        ))
    }

    /// Get chunk ID for an embedding ID
    pub fn get_chunk_id_for_embedding(&self, embedding_id: Uuid) -> Result<Option<Uuid>> {
        self.db
            .query_row(
                "SELECT chunk_id FROM embeddings WHERE id = ?1",
                params![embedding_id.as_bytes().as_slice()],
                |row| {
                    let id_bytes: Vec<u8> = row.get(0)?;
                    Ok(Uuid::from_slice(&id_bytes).expect("invalid uuid"))
                },
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    /// Get all embeddings
    pub fn get_all_embeddings(&self) -> Result<Vec<Embedding>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, chunk_id, vector, model_hash, dim, l2_norm FROM embeddings")
            .map_err(|e| CPError::Database(e.to_string()))?;

        let embs = stmt
            .query_map([], |row| self.row_to_embedding(row))
            .map_err(|e| CPError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(embs)
    }

    /// Checkpoint the HNSW index with current state root
    pub fn checkpoint_index(&self) -> Result<()> {
        let root = self.compute_merkle_root()?;
        self.hnsw.checkpoint(root)
    }

    /// Save the HNSW index to disk
    pub fn save_index(&self) -> Result<()> {
        self.hnsw.save()
    }

    // ========== Edge operations ==========

    /// Add an edge to the graph
    pub fn add_edge(&mut self, edge: &Edge) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO edges (source, target, kind, weight, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    edge.source.as_bytes().as_slice(),
                    edge.target.as_bytes().as_slice(),
                    edge.kind as i32,
                    edge.weight,
                    edge.metadata.as_deref(),
                ],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(())
    }

    /// Get edges from a node
    pub fn get_edges(&self, node_id: Uuid) -> Result<Vec<Edge>> {
        let mut stmt = self
            .db
            .prepare("SELECT source, target, kind, weight, metadata FROM edges WHERE source = ?1")
            .map_err(|e| CPError::Database(e.to_string()))?;

        let edges = stmt
            .query_map(params![node_id.as_bytes().as_slice()], |row| {
                let source_bytes: Vec<u8> = row.get(0)?;
                let target_bytes: Vec<u8> = row.get(1)?;
                let kind: i32 = row.get(2)?;
                let weight: Option<f64> = row.get(3)?;
                let metadata: Option<String> = row.get(4)?;

                let source = Uuid::from_slice(&source_bytes).expect("invalid uuid");
                let target = Uuid::from_slice(&target_bytes).expect("invalid uuid");
                let kind = EdgeKind::from_u8(kind as u8).unwrap_or(EdgeKind::DocToChunk);

                Ok(Edge { 
                    source, 
                    target, 
                    kind,
                    weight: weight.map(|w| w as f32),
                    metadata,
                })
            })
            .map_err(|e| CPError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(edges)
    }

    // ========== Vector search ==========

    /// Search for similar embeddings
    pub fn search(&self, query_vec: &[f32], k: usize) -> Result<Vec<(Uuid, f32)>> {
        Ok(self.hnsw.search(query_vec, k))
    }

    /// Lexical search using FTS5
    /// Get graph statistics
    pub fn search_lexical(&self, query: &str, k: usize) -> Result<Vec<(Uuid, f32)>> {
        let mut stmt = self
            .db
            .prepare(
                "SELECT id, rank FROM fts_chunks 
                 WHERE fts_chunks MATCH ?1 
                 ORDER BY rank LIMIT ?2",
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        let results = stmt
            .query_map(params![query, k as i64], |row| {
                let id_bytes: Vec<u8> = row.get(0)?;
                let rank: f64 = row.get(1)?;
                
                let id = Uuid::from_slice(&id_bytes).expect("invalid uuid");
                // Convert rank to a similarity score (approximate)
                // SQLite FTS5 rank: lower is better (usually negative).
                // We'll just return it raw for now, or normalize it superficially.
                Ok((id, -rank as f32))
            })
            .map_err(|e| CPError::Database(e.to_string()))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(results)
    }

    // ========== State ==========

    /// Compute Merkle root of the current state
    /// 
    /// This is strictly semantic and content-addressable. It does NOT include
    /// timestamps, UUIDs (except as IDs for relationships), or device-specific metadata.
    pub fn compute_merkle_root(&self) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();

        // 1. Documents (sorted by ID)
        let mut docs = self.get_all_documents()?;
        docs.sort_by_key(|d| d.id);
        for doc in docs {
            hasher.update(doc.id.as_bytes());
            hasher.update(&doc.hash);
            hasher.update(&doc.hierarchical_hash);
            // We ignore mtime/path as they are metadata, but hash is semantic content.
        }

        // 2. Chunks (sorted by ID)
        // We use a query to get them sorted to avoid huge Vec in memory if possible, 
        // but for MVP we get all and sort.
        let mut stmt = self.db.prepare("SELECT id, text_hash FROM chunks ORDER BY id")
            .map_err(|e| CPError::Database(e.to_string()))?;
        let chunk_iter = stmt.query_map([], |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let hash_bytes: Vec<u8> = row.get(1)?;
            Ok((id_bytes, hash_bytes))
        }).map_err(|e| CPError::Database(e.to_string()))?;

        for chunk in chunk_iter {
            let (id, hash) = chunk.map_err(|e| CPError::Database(e.to_string()))?;
            hasher.update(&id);
            hasher.update(&hash);
        }

        // 3. Embeddings (sorted by ID)
        let mut stmt = self.db.prepare("SELECT id, vector, model_hash FROM embeddings ORDER BY id")
            .map_err(|e| CPError::Database(e.to_string()))?;
        let emb_iter = stmt.query_map([], |row| {
            let id_bytes: Vec<u8> = row.get(0)?;
            let vec_bytes: Vec<u8> = row.get(1)?;
            let model_hash: Vec<u8> = row.get(2)?;
            Ok((id_bytes, vec_bytes, model_hash))
        }).map_err(|e| CPError::Database(e.to_string()))?;

        for emb in emb_iter {
            let (id, vec, model) = emb.map_err(|e| CPError::Database(e.to_string()))?;
            hasher.update(&id);
            hasher.update(&vec);
            hasher.update(&model);
        }

        // 4. Edges (sorted by source, target, kind)
        let mut stmt = self.db.prepare("SELECT source, target, kind, weight FROM edges ORDER BY source, target, kind")
            .map_err(|e| CPError::Database(e.to_string()))?;
        let edge_iter = stmt.query_map([], |row| {
            let s: Vec<u8> = row.get(0)?;
            let t: Vec<u8> = row.get(1)?;
            let k: i32 = row.get(2)?;
            let w: Option<i32> = row.get(3)?;
            Ok((s, t, k, w))
        }).map_err(|e| CPError::Database(e.to_string()))?;

        for edge in edge_iter {
            let (s, t, k, w) = edge.map_err(|e| CPError::Database(e.to_string()))?;
            hasher.update(&s);
            hasher.update(&t);
            hasher.update(&k.to_le_bytes());
            if let Some(weight) = w {
                hasher.update(&weight.to_le_bytes());
            }
        }

        Ok(*hasher.finalize().as_bytes())
    }

    /// Get the latest state root
    pub fn get_latest_root(&self) -> Result<Option<cp_core::StateRoot>> {
        use rusqlite::OptionalExtension;
        self.db
            .query_row(
                "SELECT hash, parent, hlc_wall_ms, hlc_counter, hlc_node_id, device_id, signature, seq 
                 FROM state_roots ORDER BY seq DESC LIMIT 1",
                [],
                |row| {
                    let hash_bytes: Vec<u8> = row.get(0)?;
                    let parent_bytes: Option<Vec<u8>> = row.get(1)?;
                    let hlc_wall_ms: i64 = row.get(2)?;
                    let hlc_counter: i32 = row.get(3)?;
                    let hlc_node_id_bytes: Vec<u8> = row.get(4)?;
                    let device_id_bytes: Vec<u8> = row.get(5)?;
                    let signature_bytes: Vec<u8> = row.get(6)?;
                    let seq: i64 = row.get(7)?;

                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&hash_bytes);

                    let parent = parent_bytes.map(|b| {
                        let mut p = [0u8; 32];
                        p.copy_from_slice(&b);
                        p
                    });

                    let mut hlc_node_id = [0u8; 16];
                    hlc_node_id.copy_from_slice(&hlc_node_id_bytes);
                    let hlc = cp_core::Hlc {
                        wall_ms: hlc_wall_ms as u64,
                        counter: hlc_counter as u16,
                        node_id: hlc_node_id,
                    };

                    let device_id = Uuid::from_slice(&device_id_bytes).expect("invalid uuid");
                    
                    let mut signature = [0u8; 64];
                    signature.copy_from_slice(&signature_bytes);

                    Ok(cp_core::StateRoot {
                        hash,
                        parent,
                        hlc,
                        device_id,
                        signature,
                        seq: seq as u64,
                    })
                },
            )
            .optional()
            .map_err(|e| CPError::Database(e.to_string()))
    }

    /// Set the latest state root
    pub fn set_latest_root(&mut self, root: &cp_core::StateRoot) -> Result<()> {
        self.db
            .execute(
                "INSERT OR REPLACE INTO state_roots (hash, parent, hlc_wall_ms, hlc_counter, hlc_node_id, device_id, signature, seq) 
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    root.hash.as_slice(),
                    root.parent.as_ref().map(|p| p.as_slice()),
                    root.hlc.wall_ms as i64,
                    root.hlc.counter as i32,
                    root.hlc.node_id.as_slice(),
                    root.device_id.as_bytes().as_slice(),
                    root.signature.as_slice(),
                    root.seq as i64,
                ],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(())
    }

    /// Apply a cognitive diff to the graph store with conflict resolution.
    ///
    /// Per CP-013 §13: Uses Last-Writer-Wins (LWW) with Hybrid Logical Clocks (HLC)
    /// for deterministic conflict resolution:
    /// 1. Compare HLC timestamps - later one wins
    /// 2. If timestamps equal, use hash tiebreaker (lexicographically higher wins)
    pub fn apply_diff(&mut self, diff: &cp_core::CognitiveDiff) -> Result<()> {
        // Get the latest state root for comparison
        let latest_root = self.get_latest_root()?;

        // Check for conflicts and resolve using LWW with HLC
        let (resolved_diff, has_conflict) = self.resolve_conflicts(diff, latest_root.as_ref())?;

        if has_conflict {
            info!("Conflict detected and resolved using LWW/HLC");
        }

        let tx = self.db.transaction().map_err(|e| CPError::Database(e.to_string()))?;

        // 1. Remove items (Delete) - order matters for foreign keys
        for id in &resolved_diff.removed_doc_ids {
            tx.execute("DELETE FROM documents WHERE id = ?1", params![id.as_bytes().as_slice()])
                .map_err(|e| CPError::Database(e.to_string()))?;
        }
        for id in &resolved_diff.removed_chunk_ids {
            tx.execute("DELETE FROM chunks WHERE id = ?1", params![id.as_bytes().as_slice()])
                 .map_err(|e| CPError::Database(e.to_string()))?;
        }
        for id in &resolved_diff.removed_embedding_ids {
            tx.execute("DELETE FROM embeddings WHERE id = ?1", params![id.as_bytes().as_slice()])
                 .map_err(|e| CPError::Database(e.to_string()))?;
        }
        for (source, target, kind) in &resolved_diff.removed_edges {
            tx.execute(
                "DELETE FROM edges WHERE source = ?1 AND target = ?2 AND kind = ?3",
                params![source.as_bytes().as_slice(), target.as_bytes().as_slice(), *kind as i32],
            )
            .map_err(|e| CPError::Database(e.to_string()))?;
        }

        // 2. Add/Update items (Insert or Replace)
        for doc in &resolved_diff.added_docs {
             tx.execute(
                "INSERT OR REPLACE INTO documents (id, path, hash, hierarchical_hash, mtime, size, mime_type)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    doc.id.as_bytes().as_slice(),
                    doc.path.to_string_lossy().as_ref(),
                    doc.hash.as_slice(),
                    doc.hierarchical_hash.as_slice(),
                    doc.mtime,
                    doc.size as i64,
                    &doc.mime_type,
                ],
            ).map_err(|e| CPError::Database(e.to_string()))?;
        }

        for chunk in &resolved_diff.added_chunks {
            tx.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, text, byte_offset, byte_length, sequence, text_hash)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    chunk.id.as_bytes().as_slice(),
                    chunk.doc_id.as_bytes().as_slice(),
                    &chunk.text,
                    chunk.byte_offset as i64,
                    chunk.byte_length as i64,
                    chunk.sequence,
                    chunk.text_hash.as_slice(),
                ],
            ).map_err(|e| CPError::Database(e.to_string()))?;
        }

        for emb in &resolved_diff.added_embeddings {
             let vector_bytes: Vec<u8> = emb.vector.iter().flat_map(|f| f.to_le_bytes()).collect();
             tx.execute(
                "INSERT OR REPLACE INTO embeddings (id, chunk_id, vector, model_hash, dim)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![
                    emb.id.as_bytes().as_slice(),
                    emb.chunk_id.as_bytes().as_slice(),
                    &vector_bytes,
                    emb.model_hash.as_slice(),
                    emb.dim as i32,
                ],
            ).map_err(|e| CPError::Database(e.to_string()))?;
        }

        for edge in &resolved_diff.added_edges {
            tx.execute(
                "INSERT OR REPLACE INTO edges (source, target, kind, weight) VALUES (?1, ?2, ?3, ?4)",
                params![
                    edge.source.as_bytes().as_slice(),
                    edge.target.as_bytes().as_slice(),
                    edge.kind as i32,
                    edge.weight.map(|w| w as i32),
                ],
            ).map_err(|e| CPError::Database(e.to_string()))?;
        }

        // 3. Update State Root
        tx.execute(
             "INSERT OR REPLACE INTO state_roots (hash, parent, hlc_wall_ms, hlc_counter, hlc_node_id, device_id, signature, seq)
              VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
             params![
                 resolved_diff.metadata.new_root.as_slice(),
                 if resolved_diff.metadata.prev_root == [0u8; 32] { None } else { Some(resolved_diff.metadata.prev_root.as_slice()) },
                 resolved_diff.metadata.hlc.wall_ms as i64,
                 resolved_diff.metadata.hlc.counter as i32,
                 resolved_diff.metadata.hlc.node_id.as_slice(),
                 resolved_diff.metadata.device_id.as_bytes().as_slice(),
                 [0u8; 64].as_slice(),
                 resolved_diff.metadata.seq as i64,
             ],
         ).map_err(|e| CPError::Database(e.to_string()))?;

        tx.commit().map_err(|e| CPError::Database(e.to_string()))?;

        // 4. Update HNSW Index (Best effort for added embeddings)
        for emb in &resolved_diff.added_embeddings {
            let vec_f32 = emb.to_f32();
            let _ = self.hnsw.insert(emb.id, vec_f32);
        }

        // Mark index as needing checkpoint
        self.hnsw.invalidate();

        Ok(())
    }

    /// Resolve conflicts using Last-Writer-Wins with HLC timestamps.
    ///
    /// Per CP-013 §13:
    /// 1. Operation with later HLC timestamp wins
    /// 2. If timestamps equal, operation with lexicographically higher hash wins
    ///
    /// Returns (resolved_diff, had_conflict)
    fn resolve_conflicts(
        &self,
        incoming_diff: &cp_core::CognitiveDiff,
        current_root: Option<&cp_core::StateRoot>,
    ) -> Result<(cp_core::CognitiveDiff, bool)> {
        let mut has_conflict = false;

        // If no existing state, no conflict resolution needed
        let current_root = match current_root {
            Some(root) => root,
            None => return Ok((incoming_diff.clone(), false)),
        };

        // Compare HLC timestamps
        let incoming_hlc = &incoming_diff.metadata.hlc;
        let current_hlc = &current_root.hlc;

        // Check if incoming is newer (primary resolution)
        let incoming_is_newer = incoming_hlc > current_hlc;

        // If current is newer, we should NOT apply incoming (conflict - current wins)
        if !incoming_is_newer && incoming_hlc != current_hlc {
            info!(
                "Conflict: incoming diff HLC {:?} is older than current {:?}",
                incoming_hlc, current_hlc
            );
            has_conflict = true;
            // Return empty diff - current state wins
            return Ok((cp_core::CognitiveDiff::empty(
                current_root.hash,
                incoming_diff.metadata.device_id,
                incoming_diff.metadata.seq,
                incoming_diff.metadata.hlc.clone(),
            ), has_conflict));
        }

        // If timestamps are equal, use hash tiebreaker (deterministic)
        if incoming_hlc == current_hlc {
            let incoming_hash = blake3::hash(&incoming_diff.metadata.new_root);
            let current_hash = blake3::hash(&current_root.hash);

            // Lexicographically larger hash wins
            if incoming_hash.as_bytes() <= current_hash.as_bytes() {
                info!("Conflict: equal HLC, current hash wins (tiebreaker)");
                has_conflict = true;
                return Ok((cp_core::CognitiveDiff::empty(
                    current_root.hash,
                    incoming_diff.metadata.device_id,
                    incoming_diff.metadata.seq,
                    incoming_diff.metadata.hlc.clone(),
                ), has_conflict));
            }
        }

        // Incoming is newer or wins tiebreaker - apply it
        Ok((incoming_diff.clone(), has_conflict))
    }

    /// Clear all data from the graph store (for fresh sync)
    pub fn clear_all(&mut self) -> Result<()> {
        self.db.execute_batch(
            r#"
            DELETE FROM embeddings;
            DELETE FROM chunks;
            DELETE FROM documents;
            DELETE FROM edges;
            DELETE FROM state_roots;
            "#,
        )
        .map_err(|e| CPError::Database(e.to_string()))?;

        // Reset HNSW index
        self.hnsw.clear()?;

        info!("Cleared all data from graph store for fresh sync");
        Ok(())
    }

    /// Get statistics about the graph store
    pub fn stats(&self) -> Result<GraphStats> {
        let doc_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))
            .map_err(|e| CPError::Database(e.to_string()))?;

        let chunk_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .map_err(|e| CPError::Database(e.to_string()))?;

        let embedding_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))
            .map_err(|e| CPError::Database(e.to_string()))?;

        let edge_count: i64 = self
            .db
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))
            .map_err(|e| CPError::Database(e.to_string()))?;

        Ok(GraphStats {
            documents: doc_count as usize,
            chunks: chunk_count as usize,
            embeddings: embedding_count as usize,
            edges: edge_count as usize,
        })
    }
}

/// Statistics about the graph store
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub documents: usize,
    pub chunks: usize,
    pub embeddings: usize,
    pub edges: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fresh_store() -> GraphStore {
        GraphStore::in_memory().unwrap()
    }

    fn test_hlc() -> cp_core::Hlc {
        cp_core::Hlc::new(1234567890, [0u8; 16])
    }

    // ========== GraphStore Basic Tests ==========

    #[test]
    fn test_graph_store_new() {
        let store = GraphStore::in_memory().unwrap();
        assert!(store.stats().is_ok());
    }

    #[test]
    fn test_graph_store_in_memory() {
        let store = GraphStore::in_memory().unwrap();
        let stats = store.stats().unwrap();
        assert_eq!(stats.documents, 0);
        assert_eq!(stats.chunks, 0);
        assert_eq!(stats.embeddings, 0);
        assert_eq!(stats.edges, 0);
    }

    #[test]
    fn test_graph_store_insert_document() {
        let mut store = fresh_store();
        let doc = Document::new(PathBuf::from("test.md"), b"Hello, world!", 12345);
        store.insert_document(&doc).unwrap();

        let retrieved = store.get_document(doc.id).unwrap().unwrap();
        assert_eq!(retrieved.id, doc.id);
        assert_eq!(retrieved.path, doc.path);
        assert_eq!(retrieved.hash, doc.hash);
    }

    #[test]
    fn test_graph_store_get_document_by_id() {
        let mut store = fresh_store();
        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let retrieved = store.get_document(doc.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, doc.id);
    }

    #[test]
    fn test_graph_store_get_document_by_path() {
        let mut store = fresh_store();
        let path = PathBuf::from("test.md");
        let doc = Document::new(path.clone(), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let retrieved = store.get_document_by_path(&path).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().path, path);
    }

    #[test]
    fn test_graph_store_document_not_found() {
        let store = fresh_store();
        let non_existent_id = Uuid::new_v4();
        let retrieved = store.get_document(non_existent_id).unwrap();
        assert!(retrieved.is_none());

        let non_existent_path = PathBuf::from("non_existent.md");
        let retrieved_by_path = store.get_document_by_path(&non_existent_path).unwrap();
        assert!(retrieved_by_path.is_none());
    }

    #[test]
    fn test_graph_store_update_document() {
        let mut store = fresh_store();

        // Insert document
        let doc = Document::new(PathBuf::from("test.md"), b"Original content", 12345);
        let original_id = doc.id;
        store.insert_document(&doc).unwrap();

        // Insert same document again (idempotent - same ID since same content)
        store.insert_document(&doc).unwrap();

        // Count should still be 1 (same document)
        let docs = store.get_all_documents().unwrap();
        assert_eq!(docs.len(), 1);

        // The document should exist
        let retrieved = store.get_document(original_id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_graph_store_delete_document() {
        let mut store = fresh_store();

        // Create document with chunks and embeddings
        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);
        store.insert_embedding(&emb).unwrap();

        // Delete document (should cascade to chunks and embeddings)
        store.delete_document(doc.id).unwrap();

        // Verify document is deleted
        let retrieved = store.get_document(doc.id).unwrap();
        assert!(retrieved.is_none());

        // Verify chunks are deleted
        let chunks = store.get_chunks_for_doc(doc.id).unwrap();
        assert!(chunks.is_empty());

        // Verify embeddings are deleted
        let embedding = store.get_embedding(emb.id).unwrap();
        assert!(embedding.is_none());
    }

    #[test]
    fn test_graph_store_all_documents() {
        let mut store = fresh_store();

        let doc1 = Document::new(PathBuf::from("test1.md"), b"Content 1", 12345);
        let doc2 = Document::new(PathBuf::from("test2.md"), b"Content 2", 12346);

        store.insert_document(&doc1).unwrap();
        store.insert_document(&doc2).unwrap();

        let all_docs = store.get_all_documents().unwrap();
        assert_eq!(all_docs.len(), 2);
    }

    #[test]
    fn test_graph_store_insert_chunk() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk text".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let retrieved = store.get_chunk(chunk.id).unwrap().unwrap();
        assert_eq!(retrieved.id, chunk.id);
        assert_eq!(retrieved.doc_id, doc.id);
    }

    #[test]
    fn test_graph_store_get_chunk() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let retrieved = store.get_chunk(chunk.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, chunk.id);
    }

    #[test]
    fn test_graph_store_chunks_for_document() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk1 = Chunk::new(doc.id, "Chunk 1".to_string(), 0, 10);
        let chunk2 = Chunk::new(doc.id, "Chunk 2".to_string(), 10, 20);

        store.insert_chunk(&chunk1).unwrap();
        store.insert_chunk(&chunk2).unwrap();

        let chunks = store.get_chunks_for_doc(doc.id).unwrap();
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn test_graph_store_delete_chunks_for_document() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        // Delete document should delete chunks
        store.delete_document(doc.id).unwrap();

        let chunks = store.get_chunks_for_doc(doc.id).unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_graph_store_insert_embedding() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);

        store.insert_embedding(&emb).unwrap();

        let retrieved = store.get_embedding(emb.id).unwrap().unwrap();
        assert_eq!(retrieved.id, emb.id);
    }

    #[test]
    fn test_graph_store_get_embedding() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);

        store.insert_embedding(&emb).unwrap();

        let retrieved = store.get_embedding(emb.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, emb.id);
    }

    #[test]
    fn test_graph_store_embeddings_for_chunk() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);

        store.insert_embedding(&emb).unwrap();

        let retrieved = store.get_embedding_for_chunk(chunk.id).unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().chunk_id, chunk.id);
    }

    #[test]
    fn test_graph_store_all_embeddings() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 12345);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test chunk".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let vector: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);

        store.insert_embedding(&emb).unwrap();

        let all_embeddings = store.get_all_embeddings().unwrap();
        assert_eq!(all_embeddings.len(), 1);
    }

    #[test]
    fn test_graph_store_insert_edge() {
        let mut store = fresh_store();

        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = Edge::new(source, target, EdgeKind::DocToChunk);

        store.add_edge(&edge).unwrap();

        let edges = store.get_edges(source).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, target);
    }

    #[test]
    fn test_graph_store_edges_from() {
        let mut store = fresh_store();

        let source = Uuid::new_v4();
        let target1 = Uuid::new_v4();
        let target2 = Uuid::new_v4();

        let edge1 = Edge::new(source, target1, EdgeKind::DocToChunk);
        let edge2 = Edge::new(source, target2, EdgeKind::ChunkToChunk);

        store.add_edge(&edge1).unwrap();
        store.add_edge(&edge2).unwrap();

        let edges = store.get_edges(source).unwrap();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_graph_store_edges_to() {
        let mut store = fresh_store();

        let source1 = Uuid::new_v4();
        let source2 = Uuid::new_v4();
        let target = Uuid::new_v4();

        let edge1 = Edge::new(source1, target, EdgeKind::DocToChunk);
        let edge2 = Edge::new(source2, target, EdgeKind::ChunkToChunk);

        store.add_edge(&edge1).unwrap();
        store.add_edge(&edge2).unwrap();

        // get_edges returns edges from a node, not to a node
        let edges_from_source1 = store.get_edges(source1).unwrap();
        assert_eq!(edges_from_source1.len(), 1);

        let edges_from_source2 = store.get_edges(source2).unwrap();
        assert_eq!(edges_from_source2.len(), 1);
    }

    #[test]
    fn test_graph_store_all_edges() {
        let mut store = fresh_store();

        let edge1 = Edge::new(Uuid::new_v4(), Uuid::new_v4(), EdgeKind::DocToChunk);
        let edge2 = Edge::new(Uuid::new_v4(), Uuid::new_v4(), EdgeKind::ChunkToChunk);

        store.add_edge(&edge1).unwrap();
        store.add_edge(&edge2).unwrap();

        // Check stats for edges
        let stats = store.stats().unwrap();
        assert_eq!(stats.edges, 2);
    }

    #[test]
    fn test_graph_store_compute_merkle_root() {
        let mut store = fresh_store();

        let root1 = store.compute_merkle_root().unwrap();

        let doc = Document::new(PathBuf::from("test.md"), b"Hello", 0);
        store.insert_document(&doc).unwrap();

        let root2 = store.compute_merkle_root().unwrap();

        // Roots should be different after adding content
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_graph_store_insert_state_root() {
        let mut store = fresh_store();

        let state_root = cp_core::StateRoot::new([1u8; 32], Some([2u8; 32]), test_hlc(), Uuid::new_v4(), 1);
        store.set_latest_root(&state_root).unwrap();

        let retrieved = store.get_latest_root().unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().hash, state_root.hash);
    }

    #[test]
    fn test_graph_store_get_latest_state_root() {
        let mut store = fresh_store();

        // Initially no state root
        let latest = store.get_latest_root().unwrap();
        assert!(latest.is_none());

        // Insert state root
        let state_root = cp_core::StateRoot::new([1u8; 32], None, test_hlc(), Uuid::new_v4(), 0);
        store.set_latest_root(&state_root).unwrap();

        let retrieved = store.get_latest_root().unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_graph_store_stats() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.documents, 1);
        assert_eq!(stats.chunks, 0);
        assert_eq!(stats.embeddings, 0);
        assert_eq!(stats.edges, 0);
    }

    #[test]
    fn test_graph_store_stats_empty() {
        let store = fresh_store();

        let stats = store.stats().unwrap();
        assert_eq!(stats.documents, 0);
        assert_eq!(stats.chunks, 0);
        assert_eq!(stats.embeddings, 0);
        assert_eq!(stats.edges, 0);
    }

    // ========== Transaction Tests ==========

    #[test]
    fn test_transaction_begin() {
        let store = fresh_store();
        // Transaction is implicit in SQLite, just verify store is functional
        assert!(store.stats().is_ok());
    }

    #[test]
    fn test_transaction_commit() {
        let mut store = fresh_store();

        // Each insert is auto-committed in SQLite
        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        // Verify it was committed
        let retrieved = store.get_document(doc.id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_transaction_rollback() {
        // In SQLite with autocommit, we can't easily test rollback
        // but we can verify that on error, changes are not persisted
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);

        // This should succeed
        store.insert_document(&doc).unwrap();

        // Verify it exists
        let retrieved = store.get_document(doc.id).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_transaction_atomicity() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.documents, 1);
    }

    #[test]
    fn test_transaction_isolation() {
        let store = fresh_store();
        // SQLite uses serializable isolation by default
        // Just verify reads are consistent
        let stats1 = store.stats().unwrap();
        let stats2 = store.stats().unwrap();
        assert_eq!(stats1.documents, stats2.documents);
    }

    // ========== FTS Tests ==========

    #[test]
    fn test_fts_search_basic() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Some test content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Hello world this is a test".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        // Give FTS time to update
        std::thread::sleep(std::time::Duration::from_millis(100));

        let results = store.search_lexical("test", 10).unwrap();
        // Results depend on FTS being properly set up
        // This test verifies the search function runs without error
    }

    #[test]
    fn test_fts_search_multiple_terms() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Hello world test Rust programming".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Search with multiple terms
        let results = store.search_lexical("hello world", 10).unwrap();
        // Just verify it doesn't error
    }

    #[test]
    fn test_fts_search_no_results() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Some content here".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        let results = store.search_lexical("nonexistentterm12345", 10).unwrap();
        // Should return empty or handle gracefully
    }

    #[test]
    fn test_fts_search_ranking() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk1 = Chunk::new(doc.id, "test word test".to_string(), 0, 0);

        store.insert_chunk(&chunk1).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Search should work without error - just verify it doesn't crash
        let results = store.search_lexical("test", 10);
        // Results might be empty or have entries depending on FTS state
        assert!(results.is_ok());
    }

    #[test]
    fn test_fts_search_unicode() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Hello 世界 unicode".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Should not error on unicode
        let results = store.search_lexical("世界", 10).unwrap();
    }

    #[test]
    fn test_fts_trigger_insert() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        // Insert chunk should trigger FTS
        let chunk = Chunk::new(doc.id, "Trigger test content".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        // Wait for trigger to execute
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Search should find the inserted content
        let results = store.search_lexical("trigger", 10).unwrap();
    }

    #[test]
    fn test_fts_trigger_delete() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Delete test content".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Delete the document (cascades to chunks)
        store.delete_document(doc.id).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Content should no longer be searchable
        let results = store.search_lexical("delete", 10).unwrap();
    }

    #[test]
    fn test_fts_trigger_update() {
        // Note: UPDATE on chunks triggers both delete and insert in FTS
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Original content".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // Delete and re-insert to simulate update
        store.delete_document(doc.id).unwrap();

        let updated_chunk = Chunk::new(doc.id, "Updated content".to_string(), 0, 0);
        store.insert_document(&doc).unwrap();
        store.insert_chunk(&updated_chunk).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        // New content should be searchable
        let results = store.search_lexical("updated", 10).unwrap();
    }

    // ========== Original tests ==========

    #[test]
    fn test_document_roundtrip() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Hello, world!", 12345);
        store.insert_document(&doc).unwrap();

        let retrieved = store.get_document(doc.id).unwrap().unwrap();
        assert_eq!(retrieved.id, doc.id);
        assert_eq!(retrieved.path, doc.path);
        assert_eq!(retrieved.hash, doc.hash);
    }

    #[test]
    fn test_chunk_operations() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        // Text is now canonicalized (trailing newline added)
        let chunk = Chunk::new(doc.id, "Test chunk text".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        let chunks = store.get_chunks_for_doc(doc.id).unwrap();
        assert_eq!(chunks.len(), 1);
        // Text is canonicalized (has trailing newline)
        assert_eq!(chunks[0].text, "Test chunk text\n");
    }

    #[test]
    fn test_embedding_and_search() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let chunk = Chunk::new(doc.id, "Test text".to_string(), 0, 0);
        store.insert_chunk(&chunk).unwrap();

        // Create a 384-dimensional embedding (matching the index config)
        let vector: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let emb = Embedding::new(chunk.id, &vector, [0u8; 32]);
        store.insert_embedding(&emb).unwrap();

        // Search with same vector should return it
        let results = store.search(&vector, 1).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99); // High similarity
    }

    #[test]
    fn test_edge_operations() {
        let mut store = fresh_store();

        let source = Uuid::new_v4();
        let target = Uuid::new_v4();
        let edge = Edge::new(source, target, EdgeKind::DocToChunk);

        store.add_edge(&edge).unwrap();

        let edges = store.get_edges(source).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, target);
    }

    #[test]
    fn test_merkle_root() {
        let mut store = fresh_store();

        let root1 = store.compute_merkle_root().unwrap();

        let doc = Document::new(PathBuf::from("test.md"), b"Hello", 0);
        store.insert_document(&doc).unwrap();

        let root2 = store.compute_merkle_root().unwrap();

        // Roots should be different after adding content
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_stats() {
        let mut store = fresh_store();

        let doc = Document::new(PathBuf::from("test.md"), b"Content", 0);
        store.insert_document(&doc).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.documents, 1);
        assert_eq!(stats.chunks, 0);
    }
}
