//! CP Embeddings - Local embedding generation
//!
//! Uses Candle (pure Rust) to run embedding models locally on CPU.
//! Default model: all-MiniLM-L6-v2 (384 dimensions)
//!
//! Model is downloaded on first use to ~/.cp/models/

use cp_core::{CPError, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};
use tracing::info;

/// Model manifest for embedding provenance tracking.
///
/// Per CP-001: Captures cryptographic hashes of all model components
/// to ensure deterministic and verifiable embedding generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelManifest {
    /// Human-readable model identifier (e.g., "all-MiniLM-L6-v2")
    pub model_id: String,

    /// Model version string
    pub version: String,

    /// BLAKE3 hash of the model weights file
    pub weights_hash: [u8; 32],

    /// BLAKE3 hash of the tokenizer file
    pub tokenizer_hash: [u8; 32],

    /// BLAKE3 hash of the config file
    pub config_hash: [u8; 32],

    /// Combined manifest hash: BLAKE3(weights_hash || tokenizer_hash || config_hash)
    pub manifest_hash: [u8; 32],
}

impl ModelManifest {
    /// Create a new model manifest from component hashes.
    pub fn new(
        model_id: String,
        version: String,
        weights_hash: [u8; 32],
        tokenizer_hash: [u8; 32],
        config_hash: [u8; 32],
    ) -> Self {
        // Compute combined manifest hash
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

    /// Create manifest from file paths by hashing each file.
    pub fn from_paths(
        model_id: String,
        version: String,
        weights_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        config_path: &std::path::Path,
    ) -> Result<Self> {
        let weights_hash = hash_file(weights_path)?;
        let tokenizer_hash = hash_file(tokenizer_path)?;
        let config_hash = hash_file(config_path)?;

        Ok(Self::new(
            model_id,
            version,
            weights_hash,
            tokenizer_hash,
            config_hash,
        ))
    }

    /// Get the manifest hash as a hex string.
    pub fn manifest_hash_hex(&self) -> String {
        self.manifest_hash
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect()
    }
}

/// Hash a file using BLAKE3.
fn hash_file(path: &std::path::Path) -> Result<[u8; 32]> {
    let bytes = std::fs::read(path)
        .map_err(|e| CPError::Embedding(format!("Failed to read file for hashing: {}", e)))?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

/// Default embedding model from HuggingFace
const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const MODEL_FILE: &str = "model.safetensors";
const CONFIG_FILE: &str = "config.json";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Embedding engine using Candle (pure Rust)
pub struct EmbeddingEngine {
    /// BERT model for inference
    model: BertModel,
    /// Tokenizer for text processing
    tokenizer: Tokenizer,
    /// Embedding dimension
    dim: usize,
    /// Model hash for provenance (legacy, use manifest instead)
    model_hash: [u8; 32],
    /// Full model manifest for provenance tracking
    manifest: ModelManifest,
}

impl EmbeddingEngine {
    /// Create a new embedding engine with the default model
    ///
    /// Downloads the model on first use if not already cached.
    pub fn new() -> Result<Self> {
        let model_dir = Self::get_model_dir()?;
        let model_path = model_dir.join(MODEL_FILE);
        let config_path = model_dir.join(CONFIG_FILE);
        let tokenizer_path = model_dir.join(TOKENIZER_FILE);

        // Download model if not present
        if !model_path.exists() || !config_path.exists() || !tokenizer_path.exists() {
            Self::download_model(&model_dir)?;
        }

        Self::from_path(
            model_path.to_str().unwrap(),
            config_path.to_str().unwrap(),
            tokenizer_path.to_str().unwrap(),
        )
    }

    /// Load a custom BERT model from path
    pub fn from_path(model_path: &str, config_path: &str, tokenizer_path: &str) -> Result<Self> {
        info!("Loading embedding model from {}", model_path);

        let device = Device::Cpu;

        // Create model manifest for provenance tracking
        let manifest = ModelManifest::from_paths(
            MODEL_REPO.to_string(),
            "1.0.0".to_string(),
            std::path::Path::new(model_path),
            std::path::Path::new(tokenizer_path),
            std::path::Path::new(config_path),
        )?;

        // Legacy model hash (weights only) for backwards compatibility
        let model_hash = manifest.weights_hash;

        // Load config
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| CPError::Embedding(format!("Failed to read config: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| CPError::Embedding(format!("Failed to parse config: {}", e)))?;

        // Load weights
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)
                .map_err(|e| CPError::Embedding(format!("Failed to load weights: {}", e)))?
        };

        // Load model
        let model = BertModel::load(vb, &config)
            .map_err(|e| CPError::Embedding(format!("Failed to initialize model: {}", e)))?;

        // Load tokenizer
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| CPError::Embedding(format!("Failed to load tokenizer: {}", e)))?;

        // Setup padding
        let padding = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        };
        tokenizer.with_padding(Some(padding));

        // Get embedding dimension
        let dim = config.hidden_size;

        info!("Loaded embedding model: dim={}, manifest={}", dim, manifest.manifest_hash_hex());

        Ok(Self {
            model,
            tokenizer,
            dim,
            model_hash,
            manifest,
        })
    }

    /// Get the default model directory
    fn get_model_dir() -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| CPError::Embedding("Cannot find home directory".into()))?;
        let model_dir = home.join(".cp").join("models").join("all-MiniLM-L6-v2");
        
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| CPError::Embedding(format!("Failed to create model dir: {}", e)))?;
        
        Ok(model_dir)
    }

    /// Download the model from HuggingFace
    fn download_model(model_dir: &PathBuf) -> Result<()> {
        info!("Downloading embedding model from HuggingFace...");

        let base_url = format!(
            "https://huggingface.co/{}/resolve/main",
            MODEL_REPO
        );

        // Download model.safetensors
        let model_url = format!("{}/{}", base_url, MODEL_FILE);
        let model_path = model_dir.join(MODEL_FILE);
        Self::download_file(&model_url, &model_path)?;

        // Download config.json
        let config_url = format!("{}/{}", base_url, CONFIG_FILE);
        let config_path = model_dir.join(CONFIG_FILE);
        Self::download_file(&config_url, &config_path)?;

        // Download tokenizer.json
        let tokenizer_url = format!("{}/{}", base_url, TOKENIZER_FILE);
        let tokenizer_path = model_dir.join(TOKENIZER_FILE);
        Self::download_file(&tokenizer_url, &tokenizer_path)?;

        info!("Model downloaded successfully");
        Ok(())
    }

    /// Download a file from URL to path
    fn download_file(url: &str, path: &PathBuf) -> Result<()> {
        info!("Downloading: {}", url);
        
        let response = ureq::get(url)
            .call()
            .map_err(|e| CPError::Embedding(format!("Download failed: {}", e)))?;

        let mut body = response.into_body().into_reader();
        let mut file = std::fs::File::create(path)
            .map_err(|e| CPError::Embedding(format!("Failed to create file: {}", e)))?;
        
        std::io::copy(&mut body, &mut file)
            .map_err(|e| CPError::Embedding(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Get the model hash for provenance tracking (legacy, use manifest() instead)
    pub fn model_hash(&self) -> [u8; 32] {
        self.model_hash
    }

    /// Get the full model manifest for provenance tracking
    pub fn manifest(&self) -> &ModelManifest {
        &self.manifest
    }

    /// Embed a single text
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_batch(&[text])
            .map(|v| v.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of texts
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let device = &Device::Cpu;

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| CPError::Embedding(format!("Tokenization failed: {}", e)))?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        // Convert to Tensors
        let input_ids: Vec<u32> = encodings.iter().flat_map(|e| e.get_ids().to_vec()).collect();
        let token_type_ids: Vec<u32> = encodings.iter().flat_map(|e| e.get_type_ids().to_vec()).collect();
        
        let input_ids = Tensor::from_vec(input_ids, (batch_size, seq_len), device)
            .map_err(|e| CPError::Embedding(format!("Failed to create input Tensors: {}", e)))?;
        let token_type_ids = Tensor::from_vec(token_type_ids, (batch_size, seq_len), device)
            .map_err(|e| CPError::Embedding(format!("Failed to create token type Tensors: {}", e)))?;

        // Forward pass
        let hs = self.model.forward(&input_ids, &token_type_ids, None)
            .map_err(|e| CPError::Embedding(format!("Inference failed: {}", e)))?;

        // Mean pooling
        let (_b_sz, _seq_len, _hidden_size) = hs.dims3().map_err(|e| CPError::Embedding(e.to_string()))?;
        
        // Use attention mask for proper pooling
        let mut embeddings = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mask = encodings[i].get_attention_mask();
            let hs_i = hs.get(i).map_err(|e| CPError::Embedding(e.to_string()))?;
            
            let mut sum = vec![0.0f32; self.dim];
            let mut count = 0.0f32;
            
            let data = hs_i.to_vec2::<f32>().map_err(|e| CPError::Embedding(e.to_string()))?;
            
            for (j, token_hs) in data.iter().enumerate() {
                if mask.get(j).copied().unwrap_or(0) == 1 {
                    for (k, val) in token_hs.iter().enumerate() {
                        sum[k] += val;
                    }
                    count += 1.0;
                }
            }
            
            // Normalize
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

    /// Get the dimensionality of the embeddings
    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires model download
    fn test_embed_single() {
        let engine = EmbeddingEngine::new().expect("Failed to create engine");
        let embedding = engine.embed("Hello, world!").expect("Failed to embed");
        
        assert_eq!(embedding.len(), 384);
        
        // Check normalized (unit vector)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    }

    #[test]
    #[ignore] // Requires model download
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
    #[ignore] // Requires model download
    fn test_similarity() {
        let engine = EmbeddingEngine::new().expect("Failed to create engine");
        
        let e1 = engine.embed("The cat sat on the mat").unwrap();
        let e2 = engine.embed("A cat was sitting on a mat").unwrap();
        let e3 = engine.embed("Quantum physics is complex").unwrap();
        
        // Cosine similarity (vectors are normalized)
        let sim_12: f32 = e1.iter().zip(&e2).map(|(a, b)| a * b).sum();
        let sim_13: f32 = e1.iter().zip(&e3).map(|(a, b)| a * b).sum();
        
        assert!(sim_12 > sim_13, "Similar sentences should have higher similarity");
        assert!(sim_12 > 0.8, "Similar sentences should have high similarity");
    }
}
