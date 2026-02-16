//! CP Embeddings - Local embedding generation
//!
//! Uses Candle (pure Rust) to run embedding models locally.
//! Default model: all-MiniLM-L6-v2 (384 dimensions)
//!
//! Model is embedded at compile time via include_bytes!() and cached to ~/.cache/cp/fast/

use cp_core::{CPError, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};
use tracing::info;
use lazy_static::lazy_static;

// ============================================================================
// Embedded Assets
// ============================================================================

const EMBEDDED_TOKENIZER: &[u8] = include_bytes!("../assets/tokenizer.json");
const EMBEDDED_MODEL: &[u8] = include_bytes!("../assets/model.safetensors");
const EMBEDDED_CONFIG: &[u8] = include_bytes!("../assets/config.json");

/// Model manifest for embedding provenance tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelManifest {
    pub model_id: String,
    pub version: String,
    pub weights_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub config_hash: [u8; 32],
    pub manifest_hash: [u8; 32],
}

impl ModelManifest {
    pub fn new(
        model_id: String,
        version: String,
        weights_hash: [u8; 32],
        tokenizer_hash: [u8; 32],
        config_hash: [u8; 32],
    ) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&weights_hash);
        hasher.update(&tokenizer_hash);
        hasher.update(&config_hash);
        let manifest_hash = *hasher.finalize().as_bytes();

        Self {
            model_id,
            version,
            weights_hash,
            tokenizer_hash,
            config_hash,
            manifest_hash,
        }
    }

    /// Create manifest from embedded data
    pub fn from_embedded() -> Self {
        let weights_hash = *blake3::hash(EMBEDDED_MODEL).as_bytes();
        let tokenizer_hash = *blake3::hash(EMBEDDED_TOKENIZER).as_bytes();
        let config_hash = *blake3::hash(EMBEDDED_CONFIG).as_bytes();

        Self::new(
            "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            "1.0.0".to_string(),
            weights_hash,
            tokenizer_hash,
            config_hash,
        )
    }

    pub fn manifest_hash_hex(&self) -> String {
        self.manifest_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

/// Get the model cache directory
fn get_cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| CPError::Embedding("HOME not set".into()))?;
    let cache_dir = PathBuf::from(home).join(".cache").join("cp").join("fast");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Ensure model is cached to disk (required for Candle's mmaped safetensors)
fn ensure_model_cached() -> Result<PathBuf> {
    let cache_dir = get_cache_dir()?;
    let model_path = cache_dir.join("model.safetensors");

    if !model_path.exists() {
        std::fs::write(&model_path, EMBEDDED_MODEL)?;
    }

    Ok(cache_dir)
}

/// Load tokenizer from embedded data
fn load_tokenizer() -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_bytes(EMBEDDED_TOKENIZER)
        .map_err(|e| CPError::Embedding(format!("Failed to load tokenizer: {}", e)))?;

    let padding = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    tokenizer.with_padding(Some(padding));

    Ok(tokenizer)
}

fn load_model_and_manifest() -> Result<(BertModel, ModelManifest)> {
    let cache_dir = ensure_model_cached()?;
    let model_path = cache_dir.join("model.safetensors");

    let manifest = ModelManifest::from_embedded();

    let config: Config = serde_json::from_slice(EMBEDDED_CONFIG)
        .map_err(|e| CPError::Embedding(format!("Failed to parse config: {}", e)))?;

    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)
            .map_err(|e| CPError::Embedding(format!("Failed to load weights: {}", e)))?
    };

    let model = BertModel::load(vb, &config)
        .map_err(|e| CPError::Embedding(format!("Failed to initialize model: {}", e)))?;

    Ok((model, manifest))
}

// ============================================================================
// Lazy Static Model Initialization
// ============================================================================

lazy_static! {
    static ref ENGINE: (Tokenizer, BertModel, ModelManifest, usize) = {
        let tokenizer = load_tokenizer().expect("Failed to load tokenizer");
        let (model, manifest) = load_model_and_manifest().expect("Failed to load model");
        info!("Loaded embedding model: dim=384, manifest={}", manifest.manifest_hash_hex());
        (tokenizer, model, manifest, 384)
    };
}

// ============================================================================
// Embedding Engine
// ============================================================================

pub struct EmbeddingEngine;

impl EmbeddingEngine {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn model_hash(&self) -> [u8; 32] {
        ENGINE.2.weights_hash
    }

    pub fn manifest(&self) -> &ModelManifest {
        &ENGINE.2
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_batch(&[text])
            .map(|v| v.into_iter().next().unwrap_or_default())
    }

    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (tokenizer, model, _, dim) = &*ENGINE;

        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| CPError::Embedding(format!("Tokenization failed: {}", e)))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let input_ids: Vec<u32> = encodings.iter().flat_map(|e| e.get_ids().to_vec()).collect();
        let token_type_ids: Vec<u32> = encodings.iter().flat_map(|e| e.get_type_ids().to_vec()).collect();

        let device = Device::Cpu;
        let input_tensor = Tensor::from_vec(input_ids, (batch_size, seq_len), &device)
            .map_err(|e| CPError::Embedding(format!("Failed to create input tensor: {}", e)))?;
        let token_type_tensor = Tensor::from_vec(token_type_ids, (batch_size, seq_len), &device)
            .map_err(|e| CPError::Embedding(format!("Failed to create token type tensor: {}", e)))?;

        let hidden_states = model.forward(&input_tensor, &token_type_tensor, None)
            .map_err(|e| CPError::Embedding(format!("Inference failed: {}", e)))?;

        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mask = encodings[i].get_attention_mask();
            let hs_i = hidden_states.get(i)
                .map_err(|e| CPError::Embedding(format!("Failed to get hidden state: {}", e)))?;

            let mut sum = vec![0.0f32; *dim];
            let mut count = 0.0f32;

            let data: Vec<Vec<f32>> = hs_i.to_vec2()
                .map_err(|e| CPError::Embedding(format!("Failed to convert to vec: {}", e)))?;

            for (j, token_hs) in data.iter().enumerate() {
                if mask.get(j).copied().unwrap_or(0) == 1 {
                    for (k, val) in token_hs.iter().enumerate() {
                        sum[k] += val;
                    }
                    count += 1.0;
                }
            }

            if count > 0.0 {
                let mut norm = 0.0f32;
                for v in &mut sum {
                    *v /= count;
                    norm += *v * *v;
                }
                norm = norm.sqrt();
                if norm > 0.0 {
                    for v in &mut sum {
                        *v /= norm;
                    }
                }
            }

            embeddings.push(sum);
        }

        Ok(embeddings)
    }

    pub fn dim(&self) -> usize {
        ENGINE.3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_single() {
        let engine = EmbeddingEngine::new().expect("Failed to create engine");
        let embedding = engine.embed("Hello, world!").expect("Failed to embed");

        assert_eq!(embedding.len(), 384);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    fn test_embed_batch() {
        let engine = EmbeddingEngine::new().expect("Failed to create engine");
        let embeddings = engine
            .embed_batch(&["Hello", "World", "Test"])
            .expect("Failed to embed batch");

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), 384);
        }
    }

    #[test]
    fn test_similarity() {
        let engine = EmbeddingEngine::new().expect("Failed to create engine");

        let e1 = engine.embed("The cat sat on the mat").unwrap();
        let e2 = engine.embed("A cat was sitting on a mat").unwrap();
        let e3 = engine.embed("Quantum physics is complex").unwrap();

        let sim_12: f32 = e1.iter().zip(&e2).map(|(a, b)| a * b).sum();
        let sim_13: f32 = e1.iter().zip(&e3).map(|(a, b)| a * b).sum();

        assert!(sim_12 > sim_13, "Similar sentences should have higher similarity");
        assert!(sim_12 > 0.8, "Similar sentences should have high similarity");
    }
}
