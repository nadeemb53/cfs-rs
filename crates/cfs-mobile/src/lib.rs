//! CFS Mobile - Mobile-optimized core with C FFI
//!
//! Provides a C-compatible interface for iOS and Android.

use cfs_core::{CfsError, Result};
use cfs_embeddings::EmbeddingEngine;
use cfs_graph::GraphStore;
use cfs_query::QueryEngine;
use cfs_relay_client::RelayClient;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use tokio::runtime::Runtime;
use tracing::error;
use uuid::Uuid;

static LAST_ERROR: OnceLock<Mutex<String>> = OnceLock::new();

fn get_last_error_mutex() -> &'static Mutex<String> {
    LAST_ERROR.get_or_init(|| Mutex::new(String::new()))
}

fn set_last_error(msg: &str) {
    if let Ok(mut e) = get_last_error_mutex().lock() {
        *e = msg.to_string();
    }
}

/// Opaque context for mobile operations
pub struct CfsContext {
    rt: Runtime,
    query_engine: Arc<QueryEngine>,
    graph: Arc<Mutex<GraphStore>>,
}

/// Initialize the CFS context
///
/// # Safety
/// `db_path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn cfs_init(db_path: *const c_char) -> *mut CfsContext {
    if db_path.is_null() {
        return ptr::null_mut();
    }

    let path_str = match CStr::from_ptr(db_path).to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return ptr::null_mut(),
    };

    // Initialize runtime
    let rt = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };

    // Initialize components (synchronous operations)
    // 1. Open GraphStore
    let graph = match GraphStore::open(&path_str) {
        Ok(g) => g,
        Err(e) => {
            error!("Failed to open graph store: {}", e);
            return ptr::null_mut();
        }
    };
    let graph_arc = Arc::new(Mutex::new(graph));

    // 2. Initialize Embeddings
    let embedder = match EmbeddingEngine::new() {
        Ok(e) => e,
        Err(e) => {
            error!("Failed to init embeddings: {}", e);
            return ptr::null_mut();
        }
    };
    let embedder_arc = Arc::new(embedder);

    // 3. Create QueryEngine
    let query_engine = Arc::new(QueryEngine::new(graph_arc.clone(), embedder_arc));

    let ctx = Box::new(CfsContext {
        rt,
        query_engine,
        graph: graph_arc,
    });
    
    Box::into_raw(ctx)
}

/// Sync with the relay server
///
/// Returns number of diffs applied on success, or negative error code.
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// `relay_url` must be a valid null-terminated C string.
/// `key_hex` must be a valid null-terminated C string (64 hex chars).
#[no_mangle]
pub unsafe extern "C" fn cfs_sync(
    ctx: *mut CfsContext,
    relay_url: *const c_char,
    key_hex: *const c_char,
) -> i32 {
    if ctx.is_null() || relay_url.is_null() || key_hex.is_null() {
        return -1;
    }

    let ctx = &*ctx;
    let url_str = match CStr::from_ptr(relay_url).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    
    let key_str = match CStr::from_ptr(key_hex).to_str() {
        Ok(s) => s,
        Err(_) => return -2,
    };
    
    // Parse key
    let key_bytes = match hex::decode(key_str) {
        Ok(v) => {
            if v.len() != 32 { return -4; }
            let mut k = [0u8; 32];
            k.copy_from_slice(&v);
            k
        },
        Err(_) => return -4,
    };
    
    // Create crypto engine for decryption
    let crypto = cfs_sync::CryptoEngine::new_with_seed(key_bytes);

    let client = RelayClient::new(url_str, "dummy_token");

    // Execute sync in runtime
    let res: Result<i32> = ctx.rt.block_on(async {
        // 1. Get local head
        let local_head_hash = {
            let graph = ctx.graph.lock().unwrap();
            graph.get_latest_root()?.map(|r| r.hash)
        };

        // 2. Fetch all roots from Relay
        println!("[CFS-Mobile] Fetching roots from {}", url_str);
        let remote_roots_hex = client.get_roots(None).await?;
        println!("[CFS-Mobile] Found {} roots", remote_roots_hex.len());
        let mut remote_hashes = Vec::new();
        for h in remote_roots_hex {
            if let Ok(bytes) = hex::decode(&h) {
                if bytes.len() == 32 {
                    let mut arr = [0u8; 32];
                    arr.copy_from_slice(&bytes);
                    remote_hashes.push(arr);
                }
            } else {
                println!("[CFS-Mobile] Failed to decode hex root: {}", h);
            }
        }

        // 3. Determine start point
        let start_index = if let Some(head) = local_head_hash {
            // Find our head in remote list
            if let Some(idx) = remote_hashes.iter().position(|h| *h == head) {
                idx + 1 // Start from next
            } else {
                // If the relay is empty or our head isn't there, we must start from 0
                // to ensure we have all data from the relay. 
                // Since apply_diff is idempotent (INSERT OR REPLACE), this is safe.
                println!("[CFS-Mobile] Local head not found in relay. Starting from beginning.");
                0
            }
        } else {
            0 // Start from beginning
        };

        if start_index >= remote_hashes.len() {
            println!("[CFS-Mobile] Already up to date (start_index: {}, len: {})", start_index, remote_hashes.len());
            return Ok(0);
        }

        println!("[CFS-Mobile] Syncing {} new roots...", remote_hashes.len() - start_index);
        // 2. Fetch and apply diffs
        let mut applied_count = 0;
        for root_hash in remote_hashes.iter().skip(start_index) {
            let root_hex = hex::encode(root_hash);
            // Check if we already have this root
            let root_bytes = match hex::decode(&root_hex) {
                Ok(b) => b,
                Err(e) => {
                    set_last_error(&format!("Invalid root hash: {}", e));
                    return Err(CfsError::Sync(format!("Invalid root hash: {}", e)));
                }
            };
            
            let local_has_root = {
                let graph = ctx.graph.lock().unwrap();
                // Query state_roots for exact hash
                graph.get_latest_root().map(|r| r.is_some() && r.unwrap().hash == root_bytes.as_slice()).unwrap_or(false)
            };

            if local_has_root {
                continue;
            }

            let payload = match client.get_diff(&root_hex).await {
                Ok(p) => p,
                Err(e) => {
                    set_last_error(&format!("Failed to fetch diff {}: {}", root_hex, e));
                    return Err(CfsError::Sync(format!("Failed to fetch diff {}: {}", root_hex, e)));
                }
            };

            let diff = match crypto.decrypt_diff(&payload) {
                Ok(d) => d,
                Err(e) => {
                    set_last_error(&format!("Failed to decrypt diff: {}", e));
                    return Err(CfsError::Crypto(format!("Failed to decrypt diff: {}", e)));
                }
            };

            // Before applying, verify semantic new_root in diff
            // Actually, we trust the Signed StateRoot for now in V0.
            
            {
                let mut graph = ctx.graph.lock().unwrap();
                if let Err(e) = graph.apply_diff(&diff) {
                    set_last_error(&format!("Failed to apply diff: {}", e));
                    return Err(CfsError::Database(format!("Failed to apply diff: {}", e)));
                }
                
                // Also store the StateRoot we just applied
                let state_root = cfs_core::StateRoot {
                    hash: diff.metadata.new_root,
                    parent: if diff.metadata.prev_root == [0u8; 32] { None } else { Some(diff.metadata.prev_root) },
                    timestamp: diff.metadata.timestamp,
                    device_id: diff.metadata.device_id,
                    signature: payload.signature,
                    seq: diff.metadata.seq,
                };
                if let Err(e) = graph.set_latest_root(&state_root) {
                    set_last_error(&format!("Failed to set latest root: {}", e));
                    return Err(CfsError::Database(format!("Failed to set latest root: {}", e)));
                }
            }
            applied_count += 1;
        }

        Ok(applied_count)
    });

    match res {
        Ok(count) => count,
        Err(e) => {
            println!("[CFS-Mobile] Sync Error: {}", e);
            let error_msg = e.to_string();
            set_last_error(&error_msg);
            error!("Sync failed: {}", e);
            match &e {
                CfsError::Sync(msg) => {
                    println!("[CFS-Mobile] Sync Detail: {}", msg);
                    -5
                }
                CfsError::Crypto(_) => -6,                  // Decryption/Keys error
                CfsError::Database(_) | CfsError::Io(_) => -7, // Database/File error
                CfsError::InvalidState(_) => -8,            // Sync mismatch / State error
                CfsError::Verification(_) => -9,            // Merkle/Signature check failed
                CfsError::NotFound(_) => -10,               // Diff not found on relay
                CfsError::Parse(_) => -11,                  // Diff parse error
                CfsError::Serialization(_) => -12,          // JSON parsing error
                _ => {
                    println!("[CFS-Mobile] Unmapped Error: {:?}", e);
                    -3
                }
            }
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn cfs_init_debug() {
    println!("[CFS-Mobile] Library initialized. Version: {}", String::from_utf8_lossy(CStr::from_ptr(cfs_version()).to_bytes()));
}

/// Query the knowledge graph
///
/// Returns a JSON string with search results.
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// `query` must be a valid null-terminated C string.
/// Returns a null-terminated string that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_query(
    ctx: *mut CfsContext,
    query: *const c_char,
) -> *mut c_char {
    if ctx.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let query_str = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let res: Result<Vec<SimpleSearchResult>> = (|| {
        let results = ctx.query_engine.search(query_str, 5)?;
        
        // Convert to simplified struct for JSON
        let simple_results: Vec<SimpleSearchResult> = results.into_iter().map(|r| SimpleSearchResult {
            text: r.chunk.text,
            score: r.score,
            doc_path: r.doc_path.to_string_lossy().into_owned(),
        }).collect();
        
        Ok(simple_results)
    })();

    match res {
        Ok(results) => {
            let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        },
        Err(e) => {
            error!("Query failed: {}", e);
            set_last_error(&format!("Query failed: {}", e));
            let c_str = CString::new("[]").unwrap_or_default();
            c_str.into_raw()
        }
    }
}

/// Get all chunks for a document
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// `doc_id_hex` must be a valid null-terminated C string (32 or 36 chars).
/// Returns a JSON string that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_get_chunks(
    ctx: *mut CfsContext,
    doc_id_hex: *const c_char,
) -> *mut c_char {
    if ctx.is_null() || doc_id_hex.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let doc_id_str = match CStr::from_ptr(doc_id_hex).to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };

    let doc_id = match Uuid::parse_str(doc_id_str) {
        Ok(id) => id,
        Err(_) => return ptr::null_mut(),
    };

    let res: Result<Vec<SimpleSearchResult>> = (|| {
        let results = ctx.query_engine.get_chunks_for_document(doc_id)?;
        
        // Convert to simplified struct for JSON
        let simple_results: Vec<SimpleSearchResult> = results.into_iter().map(|r| SimpleSearchResult {
            text: r.chunk.text,
            score: r.score,
            doc_path: r.doc_path.to_string_lossy().into_owned(),
        }).collect();
        
        Ok(simple_results)
    })();

    match res {
        Ok(results) => {
            let json = serde_json::to_string(&results).unwrap_or_else(|_| "[]".to_string());
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        },
        Err(e) => {
            error!("Get chunks failed: {}", e);
            let c_str = CString::new("[]").unwrap_or_default();
            c_str.into_raw()
        }
    }
}

#[derive(serde::Serialize)]
struct SimpleSearchResult {
    text: String,
    score: f32,
    doc_path: String,
}

/// Get the current state root hash as a hex string
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
/// Returns a hex string (64 chars) that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_get_state_root(ctx: *mut CfsContext) -> *mut c_char {
    if ctx.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let res = {
        let graph = ctx.graph.lock().unwrap();
        graph.get_latest_root()
    };

    match res {
        Ok(Some(root)) => {
            let hex = hex::encode(root.hash);
            let c_str = CString::new(hex).unwrap_or_default();
            c_str.into_raw()
        }
        Ok(None) => {
            let c_str = CString::new("00".repeat(32)).unwrap_or_default();
            c_str.into_raw()
        }
        Err(e) => {
            error!("Failed to get state root: {}", e);
            ptr::null_mut()
        }
    }
}

/// Free a string returned by CFS functions
///
/// # Safety
/// `s` must be a pointer returned by a CFS function or null.
#[no_mangle]
pub unsafe extern "C" fn cfs_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Free the CFS context
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init` or null.
#[no_mangle]
pub unsafe extern "C" fn cfs_free(ctx: *mut CfsContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Get the last error message
///
/// Returns a string that must be freed with `cfs_free_string`.
#[no_mangle]
pub unsafe extern "C" fn cfs_last_error() -> *mut c_char {
    let msg = if let Ok(e) = get_last_error_mutex().lock() {
        e.clone()
    } else {
        "Mutex poisoned".to_string()
    };
    let c_str = CString::new(msg).unwrap_or_default();
    c_str.into_raw()
}

/// Get graph statistics
///
/// Returns a JSON string that must be freed with `cfs_free_string`.
///
/// # Safety
/// `ctx` must be a valid pointer from `cfs_init`.
#[no_mangle]
pub unsafe extern "C" fn cfs_stats(ctx: *mut CfsContext) -> *mut c_char {
    if ctx.is_null() {
        return ptr::null_mut();
    }

    let ctx = &*ctx;
    let res = {
        let graph = ctx.graph.lock().unwrap();
        graph.stats()
    };

    match res {
        Ok(stats) => {
            let json = serde_json::json!({
                "documents": stats.documents,
                "chunks": stats.chunks,
                "embeddings": stats.embeddings,
                "edges": stats.edges,
            }).to_string();
            let c_str = CString::new(json).unwrap_or_default();
            c_str.into_raw()
        }
        Err(e) => {
            error!("Failed to get stats: {}", e);
            ptr::null_mut()
        }
    }
}

/// Get the library version
#[no_mangle]
pub extern "C" fn cfs_version() -> *const c_char {
    static VERSION: &[u8] = b"0.1.0\0";
    VERSION.as_ptr() as *const c_char
}
