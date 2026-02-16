//! CP Canonical Embeddings - ML Model-Based Deterministic Embedding Generation
//!
//! This module provides deterministic embedding generation using the MiniLM-L6-v2
//! transformer model with softfloat normalization for cross-platform consistency.
//!
//! Per CP-003:
//! - Uses sentence-transformers/all-MiniLM-L6-v2 model
//! - Embeddings stored as i16 for cross-platform consistency
//! - Deterministic quantization with dead-zone
//! - SoftFloat L2 normalization for bit-exact determinism

use thiserror::Error;
use lazy_static::lazy_static;

// ============================================================================
// Embedded Model Data (from tokenizer_data module)
// ============================================================================

/// Re-export tokenizer data for convenience
pub use tokenizer_data::{TOKENIZER_JSON, CONFIG_JSON, MODEL_DATA};

// ============================================================================
// Token Constants
// ============================================================================

pub mod tokens {
    pub const CLS: u32 = 101;
    pub const SEP: u32 = 102;
    pub const PAD: u32 = 0;
    pub const MASK: u32 = 103;
    pub const UNK: u32 = 100;
}

/// Maximum sequence length
pub const MAX_SEQ_LEN: usize = 512;

// ============================================================================
// Token Output Type
// ============================================================================

/// Output from tokenizer
#[derive(Debug, Clone)]
pub struct TokenOutput {
    pub ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub type_ids: Vec<u32>,
}

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
}

pub type Result<T> = std::result::Result<T, CanonicalError>;

// ============================================================================
// Constants
// ============================================================================

/// Embedding dimension (fixed for MiniLM-L6-v2)
pub const EMBEDDING_DIM: usize = 384;

/// Model identifier (MiniLM-L6-v2)
pub const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
pub const MODEL_VERSION: &str = "1.0.0";

/// Model hash - computed from the model identifier
pub fn model_hash() -> [u8; 32] {
    *blake3::hash(MODEL_ID.as_bytes()).as_bytes()
}

// ============================================================================
// Lazy Static Initialization for Model and Tokenizer
// ============================================================================

lazy_static! {
    /// BERT tokenizer initialized from embedded tokenizer.json
    pub static ref TOKENIZER: tokenizer_impl::BertTokenizer = {
        tokenizer_impl::BertTokenizer::new(tokenizer_data::TOKENIZER_JSON)
            .expect("Failed to initialize tokenizer")
    };

    /// MiniLM model initialized from embedded weights
    pub static ref MODEL: model::MiniLMModel = {
        let config = model::ModelConfig::from_config_json(tokenizer_data::CONFIG_JSON)
            .expect("Failed to parse config");
        model::MiniLMModel::new(tokenizer_data::MODEL_DATA, config)
            .expect("Failed to initialize model")
    };
}

// ============================================================================
// Mean Pooling Function
// ============================================================================

/// Mean pooling - takes attention mask into account for correct averaging
fn mean_pooling(model_output: &[Vec<f32>], attention_mask: &[u32]) -> Vec<f32> {
    let seq_len = model_output.len();
    if seq_len == 0 {
        return vec![0.0f32; EMBEDDING_DIM];
    }

    // Sum up all token embeddings
    let mut sum = vec![0.0f32; EMBEDDING_DIM];
    for (i, token_emb) in model_output.iter().enumerate() {
        if i < attention_mask.len() && attention_mask[i] == 1 {
            for j in 0..EMBEDDING_DIM {
                sum[j] += token_emb[j];
            }
        }
    }

    // Count valid tokens
    let valid_count = attention_mask.iter().filter(|&&x| x == 1).count() as f32;
    if valid_count == 0.0 {
        return vec![0.0f32; EMBEDDING_DIM];
    }

    // Divide by count
    for j in 0..EMBEDDING_DIM {
        sum[j] /= valid_count;
    }

    sum
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
    // Step 1: Compute L2 norm using SoftFloat-like approach
    // Use f64 for intermediate to reduce precision issues
    let norm: f64 = vector.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt();

    if norm == 0.0 {
        // Return zero vector
        return vec![0i16; vector.len()];
    }

    // Step 2: Normalize and quantize with dead-zone
    let scale = 32767.0 / norm;
    vector
        .iter()
        .map(|&v| {
            let scaled = v as f64 * scale;

            // Dead-zone: values within 0.5 of integer boundary round consistently
            let rounded = if scaled.fract().abs() < 0.5 {
                scaled.round()
            } else {
                // For values near boundaries, use deterministic tiebreaker (round toward zero)
                scaled.round_ties_even()
            };

            // Clamp to i16 range
            rounded.clamp(-32767.0, 32767.0) as i16
        })
        .collect()
}

/// Generate deterministic embedding from text using the MiniLM-L6-v2 model
///
/// This uses the full ML model for semantic embeddings with softfloat
/// normalization for deterministic cross-platform results.
pub fn embed_text(text: &str) -> Result<Embedding> {
    if text.is_empty() {
        return Ok(Embedding::from_f32(&vec![0.0f32; EMBEDDING_DIM]));
    }

    // Tokenize the input text
    let token_output = TOKENIZER.tokenize(text)?;

    // Run model inference
    let model_output = MODEL.forward(&token_output.ids, &token_output.attention_mask);

    // Apply mean pooling
    let pooled = mean_pooling(&model_output, &token_output.attention_mask);

    // Apply deterministic softfloat L2 normalization (requires fixed-size array)
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

pub mod tokenizer_data;
pub mod tokenizer_impl;
pub mod model;
pub mod softfloat;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_embedding() {
        let text = "Hello, world!";

        let emb1 = embed_text(text).unwrap();
        let emb2 = embed_text(text).unwrap();

        // Same text = same embedding
        assert_eq!(emb1.vector, emb2.vector);
        assert_eq!(emb1.id(), emb2.id());
    }

    #[test]
    fn test_different_text_different_embedding() {
        let emb1 = embed_text("Hello").unwrap();
        let emb2 = embed_text("World").unwrap();

        // Different text = different embedding
        assert_ne!(emb1.vector, emb2.vector);
        assert_ne!(emb1.id(), emb2.id());
    }

    #[test]
    fn test_embedding_dimension() {
        let emb = embed_text("test").unwrap();
        assert_eq!(emb.vector.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_quantization_deterministic() {
        let values = vec![0.1f32, 0.5, 0.9, 1.0];

        let quant1 = quantize_f32_to_i16(&values);
        let quant2 = quantize_f32_to_i16(&values);

        assert_eq!(quant1, quant2);
    }
}
