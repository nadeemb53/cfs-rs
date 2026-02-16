//! CP Canonical Embeddings - Deterministic Embedding Generation
//!
//! Per CP-010: Provides deterministic embedding generation with SoftFloat normalization.
//! Uses Candle for model loading and inference.
//!
//! - Uses intfloat/e5-small-v2 model
//! - Embeddings stored as i16 for cross-platform consistency
//! - SoftFloat L2 normalization for bit-exact determinism

use thiserror::Error;
use std::path::PathBuf;
use lazy_static::lazy_static;
use tokenizers::{Tokenizer, PaddingParams};
use candle_transformers::models::bert::DTYPE;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug)]
pub enum CanonicalError {
    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CanonicalError>;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension (fixed for E5-small-v2)
pub const EMBEDDING_DIM: usize = 384;

/// Model identifier
pub const MODEL_ID: &str = "intfloat/e5-small-v2";
pub const MODEL_VERSION: &str = "1.0.0";

/// Model hash - computed from the model identifier
pub fn model_hash() -> [u8; 32] {
    *blake3::hash(MODEL_ID.to_lowercase().as_bytes()).as_bytes()
}

/// Maximum sequence length
pub const MAX_SEQ_LEN: usize = 256;

// ============================================================================
// Model Loading
// ============================================================================

/// Get the model cache directory
fn get_model_cache_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| CanonicalError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound, "HOME not set")))?;
    let cache_dir = PathBuf::from(home)
        .join(".cache")
        .join("cp")
        .join("canonical");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Ensure model files are available (write embedded data to cache if needed)
fn ensure_model_files() -> Result<PathBuf> {
    let cache_dir = get_model_cache_dir()?;
    let model_path = cache_dir.join("model.safetensors");

    if !model_path.exists() {
        // Write embedded model to cache
        let model_data = include_bytes!("../assets/model.safetensors");
        std::fs::write(&model_path, model_data)?;
    }

    Ok(cache_dir)
}

/// Load tokenizer from embedded data
fn load_tokenizer() -> Result<Tokenizer> {
    let tokenizer_data = include_bytes!("../assets/tokenizer.json");
    let mut tokenizer = Tokenizer::from_bytes(tokenizer_data)
        .map_err(|e| CanonicalError::Tokenizer(e.to_string()))?;

    let padding = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    tokenizer.with_padding(Some(padding));

    Ok(tokenizer)
}

fn load_model() -> Result<candle_transformers::models::bert::BertModel> {
    // Ensure model is available in cache
    let cache_dir = ensure_model_files()?;
    let model_path = cache_dir.join("model.safetensors");

    // Parse config from embedded data
    let config_data = include_bytes!("../assets/config.json");
    let config: candle_transformers::models::bert::Config = serde_json::from_slice(config_data)
        .map_err(|e| CanonicalError::Inference(format!("Failed to parse config: {}", e)))?;

    // Load weights using candle
    let device = candle_core::Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_path], DTYPE, &device)
            .map_err(|e| CanonicalError::Inference(format!("Failed to load weights: {}", e)))?
    };

    let model = candle_transformers::models::bert::BertModel::load(vb, &config)
        .map_err(|e| CanonicalError::Inference(format!("Failed to initialize model: {}", e)))?;

    Ok(model)
}

// ============================================================================
// Lazy Static Model Initialization
// ============================================================================

lazy_static! {
    static ref MODEL: (Tokenizer, candle_transformers::models::bert::BertModel) = {
        let tokenizer = load_tokenizer().expect("Failed to load tokenizer");
        let model = load_model().expect("Failed to load model");
        (tokenizer, model)
    };
}

fn init_model() -> Result<&'static (Tokenizer, candle_transformers::models::bert::BertModel)> {
    Ok(&*MODEL)
}

// ============================================================================
// Mean Pooling
// ============================================================================

fn mean_pooling(hidden_states: &candle_core::Tensor, attention_mask: &[u32]) -> Result<Vec<f32>> {
    let (_batch, _seq_len, _hidden) = hidden_states.dims3()
        .map_err(|e| CanonicalError::Inference(e.to_string()))?;

    // Sum along sequence dimension
    let sum = hidden_states.sum(1)
        .map_err(|e| CanonicalError::Inference(e.to_string()))?;

    // Convert to Vec<f32>
    let sum_vec: Vec<f32> = sum.to_vec2()
        .map_err(|e| CanonicalError::Inference(e.to_string()))?
        .into_iter()
        .next()
        .unwrap_or_default();

    // Divide by attention mask count
    let count = attention_mask.iter().filter(|&&x| x == 1).count() as f32;
    if count == 0.0 {
        return Ok(vec![0.0f32; EMBEDDING_DIM]);
    }

    let result: Vec<f32> = sum_vec.iter().map(|v| v / count).collect();
    Ok(result)
}

// ============================================================================
// Embedding Types
// ============================================================================

/// Deterministic embedding vector (i16 quantized)
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub vector: Vec<i16>,
    pub model_hash: [u8; 32],
}

impl Embedding {
    /// Create embedding from raw f32 vector
    pub fn from_f32(vector: &[f32]) -> Self {
        let quantized = quantize_f32_to_i16(vector);
        Self {
            vector: quantized,
            model_hash: model_hash(),
        }
    }

    /// Get embedding as bytes (for hashing)
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.vector.len() * 2);
        for &v in &self.vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Get embedding ID (BLAKE3 of the vector bytes)
    pub fn id(&self) -> [u8; 32] {
        *blake3::hash(&self.as_bytes()).as_bytes()
    }
}

/// Quantize f32 vector to i16 with dead-zone for deterministic tiebreaking
pub fn quantize_f32_to_i16(vector: &[f32]) -> Vec<i16> {
    let norm: f64 = vector.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();

    if norm == 0.0 {
        return vec![0i16; vector.len()];
    }

    let scale = 32767.0 / norm;
    vector
        .iter()
        .map(|&v| {
            let scaled = v as f64 * scale;
            let rounded = if scaled.fract().abs() < 0.5 {
                scaled.round()
            } else {
                scaled.round_ties_even()
            };
            rounded.clamp(-32767.0, 32767.0) as i16
        })
        .collect()
}

/// Generate deterministic embedding from text using the ML model
///
/// Uses SoftFloat for deterministic L2 normalization per CP-010.
pub fn embed_text(text: &str) -> Result<Embedding> {
    if text.is_empty() {
        return Ok(Embedding::from_f32(&vec![0.0f32; EMBEDDING_DIM]));
    }

    let (tokenizer, model) = init_model()?;

    // Tokenize
    let encoding = tokenizer.encode(text, true)
        .map_err(|e| CanonicalError::Tokenizer(e.to_string()))?;

    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let token_type_ids: Vec<u32> = encoding.get_type_ids().to_vec();

    let device = candle_core::Device::Cpu;
    let input_ids_tensor = candle_core::Tensor::from_vec(
        input_ids.clone(),
        (1, input_ids.len()),
        &device,
    ).map_err(|e| CanonicalError::Inference(e.to_string()))?;

    let token_type_ids_tensor = candle_core::Tensor::from_vec(
        token_type_ids.clone(),
        (1, token_type_ids.len()),
        &device,
    ).map_err(|e| CanonicalError::Inference(e.to_string()))?;

    // Forward pass
    let hidden_states = model.forward(&input_ids_tensor, &token_type_ids_tensor, None)
        .map_err(|e| CanonicalError::Inference(e.to_string()))?;

    // Mean pooling
    let pooled = mean_pooling(&hidden_states, &attention_mask)?;

    // L2 normalize with SoftFloat
    let pooled_array: [f32; EMBEDDING_DIM] = pooled.try_into()
        .unwrap_or_else(|_| [0.0f32; EMBEDDING_DIM]);
    let normalized = softfloat::l2_normalize_softfloat(&pooled_array);

    // Quantize to i16
    let quantized = quantize_f32_to_i16(&normalized);

    Ok(Embedding {
        vector: quantized,
        model_hash: model_hash(),
    })
}

// ============================================================================
// Public Modules
// ============================================================================

pub mod softfloat;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization_deterministic() {
        let values = vec![0.1f32, 0.5, 0.9, 1.0];

        let quant1 = quantize_f32_to_i16(&values);
        let quant2 = quantize_f32_to_i16(&values);

        assert_eq!(quant1, quant2);
    }
}
