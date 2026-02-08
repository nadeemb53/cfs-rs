
use cfs_core::{CognitiveDiff, CfsError, Result, StateRoot};
use cfs_graph::GraphStore;
use cfs_relay_client::RelayClient;
use cfs_sync::{CryptoEngine};
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use tracing::info;
use blake3;

/// Manages synchronization with the relay server
pub struct SyncManager {
    relay_client: RelayClient,
    graph: Arc<Mutex<GraphStore>>,
    crypto: CryptoEngine,
    pending_diff: Mutex<CognitiveDiff>,
}

impl SyncManager {
    /// Create a new sync manager
    pub fn new(
        relay_url: &str,
        relay_token: &str,
        graph: Arc<Mutex<GraphStore>>,
        key_path: &std::path::Path,
    ) -> Result<Self> {
        // Load or generate keys
        let crypto = if key_path.exists() {
            let seed = std::fs::read(key_path).map_err(|e| CfsError::Io(e))?;
            if seed.len() != 32 {
                return Err(CfsError::Crypto("Invalid key file length".into()));
            }
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&seed);
            CryptoEngine::new_with_seed(arr)
        } else {
            // V0 Dev Default Key: 32 bytes of 0x01
            let seed = [1u8; 32];
            std::fs::write(key_path, seed).map_err(|e| CfsError::Io(e))?;
            CryptoEngine::new_with_seed(seed)
        };

        // Initialize pending diff (empty for now, essentially a "session" diff)
        // In reality, we should load pending diffs from disk or DB to persist across restarts.
        // For MVP, if we crash, we lose un-pushed changes (or regenerate them from DB scan? No).
        // Best effort: we assume DesktopApp runs long enough to push.
        let pending = CognitiveDiff::empty([0u8; 32], Uuid::new_v4(), 0);

        Ok(Self {
            relay_client: RelayClient::new(relay_url, relay_token),
            graph,
            crypto,
            pending_diff: Mutex::new(pending),
        })
    }

    /// Record a document addition
    pub fn record_add_doc(&self, doc: cfs_core::Document) {
        let mut diff = self.pending_diff.lock().unwrap();
        diff.added_docs.push(doc);
    }
    
    /// Record a chunk addition
    pub fn record_add_chunk(&self, chunk: cfs_core::Chunk) {
        let mut diff = self.pending_diff.lock().unwrap();
        diff.added_chunks.push(chunk);
    }

    /// Record an embedding addition
    pub fn record_add_embedding(&self, emb: cfs_core::Embedding) {
        let mut diff = self.pending_diff.lock().unwrap();
        diff.added_embeddings.push(emb);
    }

    /// Push pending changes to the relay
    pub async fn push(&self) -> Result<()> {
        let mut diff = {
            let mut pending = self.pending_diff.lock().unwrap();
            if pending.is_empty() {
                info!("No pending changes to sync.");
                return Ok(());
            }
            // Swap with new empty diff
            let old = pending.clone(); // efficient clone? No, huge clone!
            // TODO: optimize swap
            *pending = CognitiveDiff::empty([0u8; 32], Uuid::new_v4(), old.metadata.seq + 1);
            old
        };

        info!("Syncing {} changes...", diff.change_count());

        // 1. Get current state root from DB (this is "prev_root" for the diff)
        let (prev_root, prev_seq) = {
            let graph = self.graph.lock().unwrap();
            match graph.get_latest_root()? {
                Some(r) => (r.hash, r.seq),
                None => ([0u8; 32], 0),
            }
        };

        diff.metadata.prev_root = prev_root;
        diff.metadata.seq = prev_seq + 1;
        diff.metadata.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        diff.metadata.device_id = Uuid::new_v4(); // Or persistent ID? DiffMetadata says device_id.

        // 2. Compute new root locally
        // Phase 2 MVP: new_root = hash(prev_root + diff_hash).
        let diff_bytes = cfs_sync::serialize_diff(&diff)?;
        let diff_hash = blake3::hash(&diff_bytes);
        
        let mut hasher = blake3::Hasher::new();
        hasher.update(&prev_root);
        hasher.update(diff_hash.as_bytes());
        let new_root_hash = *hasher.finalize().as_bytes();
        diff.metadata.new_root = new_root_hash;

        // 3. Create StateRoot object and Sign it
        let signature = self.crypto.sign(&new_root_hash).to_bytes();

        let state_root = StateRoot {
            hash: new_root_hash,
            parent: if prev_root == [0u8; 32] { None } else { Some(prev_root) },
            timestamp: diff.metadata.timestamp,
            device_id: diff.metadata.device_id,
            signature,
            seq: diff.metadata.seq,
        };

        // 4. Store new root in DB
        {
            let mut graph = self.graph.lock().unwrap();
            graph.set_latest_root(&state_root)?;
        }

        // 5. Encrypt Diff
        // We need encryption key. CryptoEngine has it.
        // We need XChaCha20 key. CryptoEngine derives it from seed?
        // cfs-sync CryptoEngine uses Ed25519 seed.
        // We need a shared secret or a symmetric key for encryption.
        // cfs-sync::EncryptedPayload uses `nonce` and `ciphertext`.
        // cfs-sync::CryptoEngine has `encrypt`.
        // `encrypt(&self, plaintext) -> EncryptedPayload`.
        // It uses the ed25519 key for signing, but what for encryption?
        // 5. Encrypt Diff (Diff now contains correct new_root)
        let payload = self.crypto.encrypt_diff(&diff)?; // Encrypts and Signs

        // 6. Upload
        self.relay_client.upload_diff(payload, &state_root).await?;

        info!("Sync complete. New root: {}", state_root.hash_hex());

        Ok(())
    }
}
