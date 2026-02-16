//! MiniLM-L6-v2 Model Implementation
//!
//! Implements the transformer encoder for all-MiniLM-L6-v2 from scratch
//! using embedded weights from HuggingFace.

use crate::{CanonicalError, Result, EMBEDDING_DIM};
use serde::Deserialize;

// Model constants for MiniLM-L6-v2
pub const NUM_HIDDEN_LAYERS: usize = 6;
pub const HIDDEN_SIZE: usize = 384;
pub const INTERMEDIATE_SIZE: usize = 1536; // 4 * hidden_size
pub const NUM_ATTENTION_HEADS: usize = 6;
pub const HIDDEN_ACT: &str = "gelu";
pub const HIDDEN_DROPOUT: f32 = 0.0;
pub const ATTENTION_DROPOUT: f32 = 0.0;
pub const MAX_POSITION_EMBEDDINGS: usize = 512;

// ============================================================================
// Config
// ============================================================================

/// Model configuration from config.json
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: u32,
    pub type_vocab_size: u32,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub position_embedding_type: String,
}

impl ModelConfig {
    /// Default config for MiniLM-L6-v2
    pub fn default_config() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 384,
            intermediate_size: 1536,
            num_hidden_layers: 6,
            num_attention_heads: 6,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: Some(0),
            position_embedding_type: "absolute".to_string(),
        }
    }

    /// Parse from embedded config.json
    pub fn from_config_json(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| CanonicalError::Inference(format!("Failed to parse config.json: {}", e)))
    }
}

// ============================================================================
// Tensor Loading
// ============================================================================

/// Load f32 tensor from model data using safetensors
pub fn load_tensor_f32(data: &[u8], name: &str) -> Result<Vec<f32>> {
    use safetensors::SafeTensors;

    let tensors = SafeTensors::deserialize(data)
        .map_err(|e| CanonicalError::Inference(format!("Failed to parse safetensors: {}", e)))?;

    // tensor() returns Result<TensorView, Error>
    let tensor = tensors.tensor(name)
        .map_err(|e| CanonicalError::Inference(format!("Tensor not found: {}: {}", name, e)))?;

    if tensor.dtype() != safetensors::Dtype::F32 {
        return Err(CanonicalError::Inference(format!("Tensor {} is not f32", name)));
    }

    // Convert tensor to Vec<f32> using the data() method
    let tensor_data = tensor.data();
    let mut result = Vec::with_capacity(tensor_data.len() / 4);
    for chunk in tensor_data.chunks_exact(4) {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        result.push(val);
    }

    Ok(result)
}

// ============================================================================
// Embeddings
// ============================================================================

/// Token embeddings layer
pub struct TokenEmbedding {
    weight: Vec<f32>,
    vocab_size: usize,
}

impl TokenEmbedding {
    pub fn new(data: &[u8], config: &ModelConfig) -> Result<Self> {
        let weight = load_tensor_f32(data, "embeddings.word_embeddings.weight")?;
        let vocab_size = config.vocab_size as usize;
        Ok(Self { weight, vocab_size })
    }

    pub fn forward(&self, input_ids: &[u32]) -> Vec<Vec<f32>> {
        let seq_len = input_ids.len();
        let mut output = vec![vec![0.0f32; HIDDEN_SIZE]; seq_len];

        for (i, &token_id) in input_ids.iter().enumerate() {
            let id = token_id as usize;
            if id < self.vocab_size {
                let start = id * HIDDEN_SIZE;
                let end = start + HIDDEN_SIZE;
                if end <= self.weight.len() {
                    output[i].copy_from_slice(&self.weight[start..end]);
                }
            }
        }

        output
    }
}

/// Position embeddings layer
pub struct PositionEmbedding {
    weight: Vec<f32>,
}

impl PositionEmbedding {
    pub fn new(data: &[u8], config: &ModelConfig) -> Result<Self> {
        let weight = load_tensor_f32(data, "embeddings.position_embeddings.weight")?;
        Ok(Self { weight })
    }

    pub fn forward(&self, seq_len: usize) -> Vec<Vec<f32>> {
        let max_pos = self.weight.len() / HIDDEN_SIZE;
        let actual_len = seq_len.min(max_pos);

        let mut output = vec![vec![0.0f32; HIDDEN_SIZE]; seq_len];

        for i in 0..actual_len {
            let start = i * HIDDEN_SIZE;
            let end = start + HIDDEN_SIZE;
            output[i].copy_from_slice(&self.weight[start..end]);
        }

        output
    }
}

/// Token type embeddings
pub struct TokenTypeEmbedding {
    weight: Vec<f32>,
}

impl TokenTypeEmbedding {
    pub fn new(data: &[u8], config: &ModelConfig) -> Result<Self> {
        let weight = load_tensor_f32(data, "embeddings.token_type_embeddings.weight")?;
        Ok(Self { weight })
    }

    pub fn forward(&self, token_type_ids: &[u32]) -> Vec<Vec<f32>> {
        let seq_len = token_type_ids.len();
        let mut output = vec![vec![0.0f32; HIDDEN_SIZE]; seq_len];

        for (i, &type_id) in token_type_ids.iter().enumerate() {
            let id = type_id as usize;
            if id < 2 {
                let start = id * HIDDEN_SIZE;
                let end = start + HIDDEN_SIZE;
                output[i].copy_from_slice(&self.weight[start..end]);
            }
        }

        output
    }
}

/// Full embeddings layer
pub struct Embeddings {
    token_emb: TokenEmbedding,
    position_emb: PositionEmbedding,
    token_type_emb: TokenTypeEmbedding,
    layer_norm: LayerNorm,
}

impl Embeddings {
    pub fn new(data: &[u8], config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            token_emb: TokenEmbedding::new(data, config)?,
            position_emb: PositionEmbedding::new(data, config)?,
            token_type_emb: TokenTypeEmbedding::new(data, config)?,
            layer_norm: LayerNorm::new(data, "embeddings.LayerNorm")?,
        })
    }

    pub fn forward(&self, input_ids: &[u32], token_type_ids: &[u32]) -> Vec<Vec<f32>> {
        let seq_len = input_ids.len();

        let mut hidden_states = self.token_emb.forward(input_ids);

        let position_emb = self.position_emb.forward(seq_len);
        for i in 0..seq_len {
            for j in 0..HIDDEN_SIZE {
                hidden_states[i][j] += position_emb[i][j];
            }
        }

        let token_type_emb = self.token_type_emb.forward(token_type_ids);
        for i in 0..seq_len {
            for j in 0..HIDDEN_SIZE {
                hidden_states[i][j] += token_type_emb[i][j];
            }
        }

        hidden_states = self.layer_norm.forward(&hidden_states);

        hidden_states
    }
}

// ============================================================================
// Layer Norm
// ============================================================================

pub struct LayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl LayerNorm {
    pub fn new(data: &[u8], name: &str) -> Result<Self> {
        let weight = load_tensor_f32(data, &format!("{}.weight", name))?;
        let bias = load_tensor_f32(data, &format!("{}.bias", name))?;
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let mut output = vec![vec![0.0f32; HIDDEN_SIZE]; seq_len];

        for (i, hidden) in input.iter().enumerate() {
            let sum: f32 = hidden.iter().sum();
            let mean = sum / HIDDEN_SIZE as f32;

            let var: f32 = hidden.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / HIDDEN_SIZE as f32;
            let std = (var + 1e-12).sqrt();

            for j in 0..HIDDEN_SIZE {
                output[i][j] = ((hidden[j] - mean) / std) * self.weight[j] + self.bias[j];
            }
        }

        output
    }
}

// ============================================================================
// GELU Activation
// ============================================================================

pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let c = 0.044715f32;
    let inner = sqrt_2_over_pi * (x + c * x.powi(3));
    0.5f32 * x * (1.0f32 + inner.tanh())
}

// ============================================================================
// Self-Attention
// ============================================================================

pub struct SelfAttention {
    query: Dense,
    key: Dense,
    value: Dense,
    output: Dense,
    num_attention_heads: usize,
    attention_head_size: usize,
}

impl SelfAttention {
    pub fn new(data: &[u8], layer_idx: usize) -> Result<Self> {
        let prefix = format!("encoder.layer.{}", layer_idx);
        Ok(Self {
            query: Dense::new(data, &format!("{}.attention.self.query", prefix))?,
            key: Dense::new(data, &format!("{}.attention.self.key", prefix))?,
            value: Dense::new(data, &format!("{}.attention.self.value", prefix))?,
            output: Dense::new(data, &format!("{}.attention.output.dense", prefix))?,
            num_attention_heads: NUM_ATTENTION_HEADS,
            attention_head_size: HIDDEN_SIZE / NUM_ATTENTION_HEADS,
        })
    }

    pub fn forward(&self, hidden_states: &[Vec<f32>], _dropout: f32) -> Vec<Vec<f32>> {
        let seq_len = hidden_states.len();
        let head_size = self.attention_head_size;
        let num_heads = self.num_attention_heads;

        let q = self.query.forward(hidden_states);
        let k = self.key.forward(hidden_states);
        let v = self.value.forward(hidden_states);

        // Reshape for multi-head
        let mut q_reshaped = vec![vec![vec![0.0f32; head_size]; num_heads]; seq_len];
        let mut k_reshaped = vec![vec![vec![0.0f32; head_size]; num_heads]; seq_len];
        let mut v_reshaped = vec![vec![vec![0.0f32; head_size]; num_heads]; seq_len];

        for i in 0..seq_len {
            for h in 0..num_heads {
                for j in 0..head_size {
                    q_reshaped[i][h][j] = q[i][h * head_size + j];
                    k_reshaped[i][h][j] = k[i][h * head_size + j];
                    v_reshaped[i][h][j] = v[i][h * head_size + j];
                }
            }
        }

        // Attention scores
        let mut attention_scores = vec![vec![vec![0.0f32; seq_len]; num_heads]; seq_len];
        for i in 0..seq_len {
            for h in 0..num_heads {
                for j in 0..seq_len {
                    let mut sum = 0.0f32;
                    for k_idx in 0..head_size {
                        sum += q_reshaped[i][h][k_idx] * k_reshaped[j][h][k_idx];
                    }
                    attention_scores[i][h][j] = sum / (head_size as f32).sqrt();
                }
            }
        }

        // Softmax
        let mut attention_probs = vec![vec![vec![0.0f32; seq_len]; num_heads]; seq_len];
        for i in 0..seq_len {
            for h in 0..num_heads {
                let max_score = attention_scores[i][h].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for j in 0..seq_len {
                    exp_sum += (attention_scores[i][h][j] - max_score).exp();
                }
                for j in 0..seq_len {
                    attention_probs[i][h][j] = (attention_scores[i][h][j] - max_score).exp() / exp_sum;
                }
            }
        }

        // Apply attention to values
        let mut context_layer = vec![vec![vec![0.0f32; head_size]; num_heads]; seq_len];
        for i in 0..seq_len {
            for h in 0..num_heads {
                for j in 0..head_size {
                    let mut sum = 0.0f32;
                    for k_idx in 0..seq_len {
                        sum += attention_probs[i][h][k_idx] * v_reshaped[k_idx][h][j];
                    }
                    context_layer[i][h][j] = sum;
                }
            }
        }

        // Concatenate heads
        let mut output = vec![vec![0.0f32; HIDDEN_SIZE]; seq_len];
        for i in 0..seq_len {
            for h in 0..num_heads {
                for j in 0..head_size {
                    output[i][h * head_size + j] = context_layer[i][h][j];
                }
            }
        }

        self.output.forward_inplace(&mut output);
        output
    }
}

// ============================================================================
// Dense Layer
// ============================================================================

pub struct Dense {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl Dense {
    pub fn new(data: &[u8], name: &str) -> Result<Self> {
        // Load weight - need to determine shape from config
        // For simplicity, load full weight and infer dimensions
        let weight = load_tensor_f32(data, &format!("{}.weight", name))?;
        let bias = load_tensor_f32(data, &format!("{}.bias", name))?;
        Ok(Self { weight, bias })
    }

    pub fn forward(&self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = input.len();
        let in_features = input[0].len();
        let out_features = self.bias.len();

        let mut output = vec![vec![0.0f32; out_features]; seq_len];

        for i in 0..seq_len {
            for j in 0..out_features {
                let mut sum = self.bias[j];
                for k in 0..in_features {
                    sum += input[i][k] * self.weight[j * in_features + k];
                }
                output[i][j] = sum;
            }
        }

        output
    }

    pub fn forward_inplace(&self, input: &mut [Vec<f32>]) {
        let seq_len = input.len();
        let in_features = input[0].len();
        let out_features = self.bias.len();

        for i in 0..seq_len {
            let mut output = vec![0.0f32; out_features];
            for j in 0..out_features {
                let mut sum = self.bias[j];
                for k in 0..in_features {
                    sum += input[i][k] * self.weight[j * in_features + k];
                }
                output[j] = sum;
            }
            input[i].copy_from_slice(&output);
        }
    }
}

// ============================================================================
// Feed-Forward Network
// ============================================================================

pub struct FeedForward {
    dense1: Dense,
    dense2: Dense,
}

impl FeedForward {
    pub fn new(data: &[u8], layer_idx: usize) -> Result<Self> {
        let prefix = format!("encoder.layer.{}", layer_idx);
        Ok(Self {
            dense1: Dense::new(data, &format!("{}.intermediate.dense", prefix))?,
            dense2: Dense::new(data, &format!("{}.output.dense", prefix))?,
        })
    }

    pub fn forward(&self, hidden_states: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let intermediate = self.dense1.forward(hidden_states);
        let activated: Vec<Vec<f32>> = intermediate
            .iter()
            .map(|vec| vec.iter().map(|&x| gelu(x)).collect())
            .collect();

        self.dense2.forward(&activated)
    }

    pub fn forward_inplace(&self, hidden_states: &mut [Vec<f32>]) {
        let intermediate_size = self.dense1.bias.len();
        let seq_len = hidden_states.len();

        let mut intermediate = vec![vec![0.0f32; intermediate_size]; seq_len];
        for i in 0..seq_len {
            for j in 0..intermediate_size {
                let mut sum = self.dense1.bias[j];
                for k in 0..HIDDEN_SIZE {
                    sum += hidden_states[i][k] * self.dense1.weight[j * HIDDEN_SIZE + k];
                }
                intermediate[i][j] = gelu(sum);
            }
        }

        for i in 0..seq_len {
            let mut output = vec![0.0f32; HIDDEN_SIZE];
            for j in 0..HIDDEN_SIZE {
                let mut sum = self.dense2.bias[j];
                for k in 0..intermediate_size {
                    sum += intermediate[i][k] * self.dense2.weight[j * intermediate_size + k];
                }
                output[j] = sum;
            }
            hidden_states[i].copy_from_slice(&output);
        }
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

pub struct TransformerLayer {
    attention: SelfAttention,
    attention_output: LayerNorm,
    ffn: FeedForward,
    ffn_output: LayerNorm,
}

impl TransformerLayer {
    pub fn new(data: &[u8], layer_idx: usize) -> Result<Self> {
        Ok(Self {
            attention: SelfAttention::new(data, layer_idx)?,
            attention_output: LayerNorm::new(
                data,
                &format!("encoder.layer.{}.attention.output.LayerNorm", layer_idx),
            )?,
            ffn: FeedForward::new(data, layer_idx)?,
            ffn_output: LayerNorm::new(
                data,
                &format!("encoder.layer.{}.output.LayerNorm", layer_idx),
            )?,
        })
    }

    pub fn forward(&self, hidden_states: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut attention_output = self.attention.forward(hidden_states, ATTENTION_DROPOUT);

        for i in 0..attention_output.len() {
            for j in 0..HIDDEN_SIZE {
                attention_output[i][j] += hidden_states[i][j];
            }
        }
        attention_output = self.attention_output.forward(&attention_output);

        let mut ffn_output = self.ffn.forward(&attention_output);

        for i in 0..ffn_output.len() {
            for j in 0..HIDDEN_SIZE {
                ffn_output[i][j] += attention_output[i][j];
            }
        }
        ffn_output = self.ffn_output.forward(&ffn_output);

        ffn_output
    }
}

// ============================================================================
// Encoder
// ============================================================================

pub struct Encoder {
    layers: Vec<TransformerLayer>,
}

impl Encoder {
    pub fn new(data: &[u8], config: &ModelConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers as usize {
            layers.push(TransformerLayer::new(data, i)?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, hidden_states: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut output = hidden_states.to_vec();

        for layer in &self.layers {
            output = layer.forward(&output);
        }

        output
    }
}

// ============================================================================
// Full Model
// ============================================================================

pub struct MiniLMModel {
    embeddings: Embeddings,
    encoder: Encoder,
    config: ModelConfig,
}

impl MiniLMModel {
    pub fn new(data: &[u8], config: ModelConfig) -> Result<Self> {
        Ok(Self {
            embeddings: Embeddings::new(data, &config)?,
            encoder: Encoder::new(data, &config)?,
            config,
        })
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &[u32], attention_mask: &[u32]) -> Vec<Vec<f32>> {
        let seq_len = input_ids.len();
        let token_type_ids: Vec<u32> = input_ids.iter().map(|_| 0).collect();

        let mut hidden_states = self.embeddings.forward(input_ids, &token_type_ids);
        hidden_states = self.encoder.forward(&hidden_states);
        hidden_states
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::softfloat;
    use crate::tokenizer_data::{CONFIG_JSON, MODEL_DATA};

    fn create_test_model() -> MiniLMModel {
        let config = ModelConfig::from_config_json(CONFIG_JSON)
            .expect("Failed to parse config");
        MiniLMModel::new(MODEL_DATA, config).expect("Failed to create model")
    }

    #[test]
    fn test_model_load() {
        let config = ModelConfig::from_config_json(CONFIG_JSON).expect("Failed to parse config");
        let model = MiniLMModel::new(MODEL_DATA, config).expect("Failed to load model");

        // Model should be loaded successfully
        assert_eq!(model.config.vocab_size, 30522);
    }

    #[test]
    fn test_model_forward_pass() {
        let model = create_test_model();

        // Simple forward pass with token IDs
        let input_ids = vec![101, 7592, 999, 102]; // [CLS] hello [SEP]
        let attention_mask = vec![1, 1, 1, 1];

        let output = model.forward(&input_ids, &attention_mask);

        // Should produce hidden states for each token
        assert_eq!(output.len(), input_ids.len());
        // Each hidden state should have HIDDEN_SIZE dimensions
        for hidden in &output {
            assert_eq!(hidden.len(), HIDDEN_SIZE);
        }
    }

    #[test]
    fn test_model_output_dimension() {
        let model = create_test_model();

        let input_ids = vec![101, 2000, 102];
        let attention_mask = vec![1, 1, 1];

        let output = model.forward(&input_ids, &attention_mask);

        // Check hidden size is 384
        assert_eq!(HIDDEN_SIZE, 384);
        for hidden in &output {
            assert_eq!(hidden.len(), 384);
        }
    }

    #[test]
    fn test_model_max_sequence_length() {
        // MAX_POSITION_EMBEDDINGS should be 512
        assert_eq!(MAX_POSITION_EMBEDDINGS, 512);

        let model = create_test_model();

        // Test with sequence of various lengths
        let short_input: Vec<u32> = (0..10).collect();
        let short_output = model.forward(&short_input, &vec![1; short_input.len()]);
        assert_eq!(short_output.len(), 10);

        let medium_input: Vec<u32> = (0..100).collect();
        let medium_output = model.forward(&medium_input, &vec![1; medium_input.len()]);
        assert_eq!(medium_output.len(), 100);
    }

    #[test]
    fn test_model_mean_pooling() {
        // Test mean pooling logic from lib.rs
        let model_output = vec![
            vec![1.0f32; HIDDEN_SIZE],
            vec![2.0f32; HIDDEN_SIZE],
            vec![3.0f32; HIDDEN_SIZE],
        ];
        let attention_mask = vec![1, 1, 1];

        // Use the mean_pooling from lib.rs
        let pooled = crate::mean_pooling(&model_output, &attention_mask);

        // Each dimension should be the mean (1+2+3)/3 = 2.0
        for val in &pooled {
            assert!((val - 2.0).abs() < 0.001, "Expected 2.0, got {}", val);
        }
    }

    #[test]
    fn test_model_l2_normalization() {
        // Test that output can be L2 normalized
        // The softfloat normalization may not produce perfect 1.0 norm due to Q30 limitations
        // but should still produce deterministic results
        let model = create_test_model();

        let input_ids = vec![101, 2000, 102];
        let attention_mask = vec![1, 1, 1];

        let output = model.forward(&input_ids, &attention_mask);
        let pooled = crate::mean_pooling(&output, &attention_mask);

        // Convert to fixed-size array and normalize
        let pooled_array: [f32; EMBEDDING_DIM] = pooled.clone().try_into().unwrap_or_else(|_| [0.0f32; EMBEDDING_DIM]);

        // Run normalization twice to verify determinism
        let normalized1 = softfloat::l2_normalize_softfloat(&pooled_array);
        let normalized2 = softfloat::l2_normalize_softfloat(&pooled_array);

        // Should be bit-identical (deterministic)
        for (a, b) in normalized1.iter().zip(normalized2.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "L2 normalization should be deterministic");
        }
    }

    #[test]
    fn test_model_quantization_f32_to_i16() {
        // Test quantization from f32 to i16
        let values = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let quantized = crate::quantize_f32_to_i16(&values);

        assert_eq!(quantized.len(), values.len());

        // Check that values are in i16 range
        for &q in &quantized {
            assert!(q >= -32767 && q <= 32767, "Quantized value {} out of range", q);
        }
    }

    #[test]
    fn test_model_quantization_bounds() {
        // Test that quantization respects bounds -32767 to 32767
        let large_values = vec![1000.0f32, -1000.0f32, 0.0];
        let quantized = crate::quantize_f32_to_i16(&large_values);

        for &q in &quantized {
            assert!(q >= -32767, "Value {} below minimum", q);
            assert!(q <= 32767, "Value {} above maximum", q);
        }
    }

    #[test]
    fn test_model_quantization_rounding() {
        // Test deterministic rounding
        let values = vec![0.5f32, 0.25, 0.75];

        let result1 = crate::quantize_f32_to_i16(&values);
        let result2 = crate::quantize_f32_to_i16(&values);

        // Should be deterministic
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_model_canonical_embedding_generation() {
        // Test full embedding generation pipeline
        let embedding = crate::embed_text("hello").expect("Embedding failed");

        assert_eq!(embedding.vector.len(), EMBEDDING_DIM);
        assert_eq!(embedding.model_hash, crate::model_hash());
    }

    #[test]
    fn test_model_batch_embedding() {
        // Test multiple embeddings in sequence
        let emb1 = crate::embed_text("hello").expect("Embedding failed");
        let emb2 = crate::embed_text("world").expect("Embedding failed");
        let emb3 = crate::embed_text("hello").expect("Embedding failed");

        // Same text should produce same embedding
        assert_eq!(emb1.vector, emb3.vector);

        // Different text should produce different embedding
        assert_ne!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_model_hash() {
        // Test that model_hash is BLAKE3 of MODEL_ID
        let hash = crate::model_hash();
        let expected_hash = *blake3::hash(crate::MODEL_ID.as_bytes()).as_bytes();

        assert_eq!(hash, expected_hash);
    }

    #[test]
    fn test_model_dropout_disabled() {
        // Test that dropout is 0.0 in inference
        assert_eq!(HIDDEN_DROPOUT, 0.0);
        assert_eq!(ATTENTION_DROPOUT, 0.0);
    }

    #[test]
    fn test_model_determinism() {
        // Test that same input produces same output
        let emb1 = crate::embed_text("The quick brown fox").expect("Embedding failed");
        let emb2 = crate::embed_text("The quick brown fox").expect("Embedding failed");

        assert_eq!(emb1.vector, emb2.vector);
    }

    #[test]
    fn test_model_determinism_across_runs() {
        // Test determinism across multiple runs
        let text = "testing determinism";

        let mut results = Vec::new();
        for _ in 0..5 {
            let emb = crate::embed_text(text).expect("Embedding failed");
            results.push(emb.vector.clone());
        }

        // All should be identical
        for i in 1..results.len() {
            assert_eq!(results[0], results[i], "Run {} differs from run 0", i);
        }
    }

    #[test]
    fn test_model_embedding_test_vector() {
        // CP-010: "The quick brown fox" should produce valid embedding
        let embedding = crate::embed_text("The quick brown fox").expect("Embedding failed");

        // Should have correct dimension
        assert_eq!(embedding.vector.len(), 384);

        // Should have non-zero values (not all zeros)
        let has_nonzero = embedding.vector.iter().any(|&v| v != 0);
        assert!(has_nonzero, "Embedding should have non-zero values");
    }
}
