//! Embedding node representing a vector derived from a chunk
//!
//! Stores the vector in i16 format (quantized) for determinism and storage efficiency.
//! Per CP-010: unit vectors are scaled by 32767, quantized with round_ties_even.
//! Per CP-003 §4.3: canonical operations use integer math.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An embedding vector derived from a chunk
///
/// Stores the vector in i16 format (quantized) for determinism and storage efficiency.
/// The embedding ID is derived deterministically via BLAKE3-16 from chunk_id + model_hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding {
    /// Unique identifier for this embedding (BLAKE3-16 of chunk_id || model_hash)
    pub id: Uuid,

    /// Parent chunk ID
    pub chunk_id: Uuid,

    /// The embedding vector (i16 quantized, scale = 32767)
    pub vector: Vec<i16>,

    /// Hash of the model weights used to generate this embedding
    pub model_hash: [u8; 32],

    /// Dimensionality of the vector
    pub dim: u16,

    /// Precomputed L2 norm of the quantized vector (for similarity computation)
    /// Per CP-001: stored for efficient cosine similarity without recomputation
    pub l2_norm: f32,
}

impl Embedding {
    /// Create a new embedding from an f32 vector.
    ///
    /// Per CP-010 §3.4-3.5:
    /// 1. Normalize f32 vector to unit length
    /// 2. Quantize to i16 with round_ties_even (scale = 32767)
    ///
    /// The embedding ID is deterministic: BLAKE3-16(chunk_id || model_hash).
    pub fn new(chunk_id: Uuid, vector_f32: &[f32], model_hash: [u8; 32]) -> Self {
        // 1. Normalize f32 vector to unit length
        let normalized = normalize_l2(vector_f32);

        // 2. Quantize to i16 (scale by 32767, round_ties_even per CP-003 §4.3)
        let quantized: Vec<i16> = normalized
            .iter()
            .map(|&v| quantize_f32_to_i16(v))
            .collect();

        let dim = quantized.len() as u16;

        // Compute L2 norm of quantized vector
        let l2_norm = compute_l2_norm(&quantized);

        // Deterministic ID: BLAKE3-16(chunk_id || model_hash)
        let id_bytes = crate::id::generate_composite_id(&[
            chunk_id.as_bytes(),
            &model_hash,
        ]);
        let id = Uuid::from_bytes(id_bytes);

        Self {
            id,
            chunk_id,
            vector: quantized,
            model_hash,
            dim,
            l2_norm,
        }
    }

    /// Create an embedding directly from pre-quantized i16 values.
    ///
    /// Used when loading from storage where quantization already occurred.
    pub fn from_quantized(
        chunk_id: Uuid,
        vector: Vec<i16>,
        model_hash: [u8; 32],
    ) -> Self {
        let dim = vector.len() as u16;
        let l2_norm = compute_l2_norm(&vector);
        let id_bytes = crate::id::generate_composite_id(&[
            chunk_id.as_bytes(),
            &model_hash,
        ]);
        let id = Uuid::from_bytes(id_bytes);

        Self {
            id,
            chunk_id,
            vector,
            model_hash,
            dim,
            l2_norm,
        }
    }

    /// Create an embedding from pre-quantized values with a precomputed L2 norm.
    ///
    /// Used when loading from storage where the norm was already stored.
    pub fn from_quantized_with_norm(
        chunk_id: Uuid,
        vector: Vec<i16>,
        model_hash: [u8; 32],
        l2_norm: f32,
    ) -> Self {
        let dim = vector.len() as u16;
        let id_bytes = crate::id::generate_composite_id(&[
            chunk_id.as_bytes(),
            &model_hash,
        ]);
        let id = Uuid::from_bytes(id_bytes);

        Self {
            id,
            chunk_id,
            vector,
            model_hash,
            dim,
            l2_norm,
        }
    }

    /// Convert the quantized vector back to f32 (approximate).
    pub fn to_f32(&self) -> Vec<f32> {
        self.vector.iter().map(|&v| v as f32 / 32767.0).collect()
    }

    /// Compute integer dot product between this embedding and another i16 vector.
    ///
    /// Per CP-003 §4.5: all similarity computations use integer math.
    /// Returns i64 to avoid overflow (384 dims * 32767^2 fits in i64).
    pub fn integer_dot_product(&self, other: &[i16]) -> i64 {
        if self.vector.len() != other.len() {
            return 0;
        }

        self.vector
            .iter()
            .zip(other.iter())
            .map(|(&a, &b)| (a as i64) * (b as i64))
            .sum()
    }

    /// Compute the squared L2 norm of the quantized vector (integer).
    ///
    /// This avoids sqrt and floating-point entirely.
    pub fn norm_squared(&self) -> i64 {
        self.vector
            .iter()
            .map(|&v| (v as i64) * (v as i64))
            .sum()
    }

    /// Compute L2 norm as f32 (for display/diagnostics only, not canonical).
    pub fn norm_f32(&self) -> f32 {
        (self.norm_squared() as f64).sqrt() as f32
    }

    /// Compute cosine similarity using integer math.
    ///
    /// Returns f32 for convenience, but the dot product and norms are
    /// computed entirely in integer arithmetic first.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.vector.len() != other.vector.len() {
            return 0.0;
        }

        let dot = self.integer_dot_product(&other.vector);
        let norm_a = self.norm_squared();
        let norm_b = other.norm_squared();

        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }

        // Final division: integer results → f32
        // dot / sqrt(norm_a * norm_b)
        let denom = ((norm_a as f64) * (norm_b as f64)).sqrt();
        (dot as f64 / denom) as f32
    }
}

/// Normalize vector to unit length
fn normalize_l2(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|v| v / norm).collect()
}

/// Compute L2 norm of quantized i16 vector
fn compute_l2_norm(vector: &[i16]) -> f32 {
    let sum_sq: i64 = vector.iter().map(|&v| (v as i64) * (v as i64)).sum();
    (sum_sq as f64).sqrt() as f32
}

/// Quantize f32 (-1.0 to 1.0) to i16 (-32767 to 32767)
///
/// Per CP-003 §4.3: uses round_ties_even for deterministic rounding.
/// Dead-zone: values in (-1e-7, 1e-7) map to 0.
fn quantize_f32_to_i16(val: f32) -> i16 {
    // Dead-zone per CP-003 §4.3
    if val.abs() < 1e-7 {
        return 0;
    }
    let scaled = val * 32767.0;
    let rounded = scaled.round_ties_even();
    rounded.clamp(-32767.0, 32767.0) as i16
}

impl PartialEq for Embedding {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.chunk_id == other.chunk_id
            && self.model_hash == other.model_hash
            && self.dim == other.dim
    }
}

impl Eq for Embedding {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_id_is_blake3_not_uuid_v5() {
        let chunk_id = Uuid::from_bytes([42u8; 16]);
        let model_hash = [1u8; 32];
        let vector = vec![1.0, 0.0];

        let emb = Embedding::new(chunk_id, &vector, model_hash);

        // Verify ID matches BLAKE3-16(chunk_id || model_hash)
        let expected = crate::id::generate_composite_id(&[
            chunk_id.as_bytes(),
            &model_hash,
        ]);
        assert_eq!(emb.id.as_bytes(), &expected);
    }

    #[test]
    fn test_embedding_creation_quantized() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        // [1.0, 0.0] -> Normalized [1.0, 0.0] -> Quantized [32767, 0]
        let vector = vec![1.0, 0.0];
        let model_hash = [0u8; 32];

        let emb = Embedding::new(chunk_id, &vector, model_hash);

        assert_eq!(emb.vector[0], 32767);
        assert_eq!(emb.vector[1], 0);
        assert!((emb.norm_f32() - 32767.0).abs() < 1.0);
    }

    #[test]
    fn test_quantize_round_ties_even() {
        // 0.5 * 32767 = 16383.5 → should round to 16384 (even)
        let result = quantize_f32_to_i16(0.5);
        // round_ties_even(16383.5) = 16384
        assert_eq!(result, 16384);
    }

    #[test]
    fn test_quantize_dead_zone() {
        assert_eq!(quantize_f32_to_i16(0.0), 0);
        assert_eq!(quantize_f32_to_i16(1e-8), 0);  // Below dead-zone threshold
        assert_eq!(quantize_f32_to_i16(-1e-8), 0); // Below dead-zone threshold
    }

    #[test]
    fn test_integer_dot_product() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        let model_hash = [0u8; 32];

        let emb = Embedding::from_quantized(chunk_id, vec![100, 200, 300], model_hash);
        let other = vec![1i16, 2, 3];

        // 100*1 + 200*2 + 300*3 = 100 + 400 + 900 = 1400
        assert_eq!(emb.integer_dot_product(&other), 1400);
    }

    #[test]
    fn test_cosine_similarity() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        let model_hash = [0u8; 32];

        let emb1 = Embedding::new(chunk_id, &[1.0, 0.0], model_hash);
        let emb2 = Embedding::new(chunk_id, &[1.0, 0.0], model_hash);
        let emb3 = Embedding::new(chunk_id, &[0.0, 1.0], model_hash); // Orthogonal
        let emb4 = Embedding::new(chunk_id, &[-1.0, 0.0], model_hash); // Opposite

        assert!((emb1.cosine_similarity(&emb2) - 1.0).abs() < 0.01);
        assert!(emb1.cosine_similarity(&emb3).abs() < 0.01);
        assert!((emb1.cosine_similarity(&emb4) + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_embedding_id_determinism() {
        let chunk_id = Uuid::from_bytes([42u8; 16]);
        let model_hash = [7u8; 32];
        let vector = vec![0.5, -0.3, 0.8];

        let emb1 = Embedding::new(chunk_id, &vector, model_hash);
        let emb2 = Embedding::new(chunk_id, &vector, model_hash);
        assert_eq!(emb1.id, emb2.id);
    }

    #[test]
    fn test_from_quantized() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        let model_hash = [0u8; 32];
        let vec = vec![32767i16, 0, -32767];

        let emb = Embedding::from_quantized(chunk_id, vec.clone(), model_hash);
        assert_eq!(emb.vector, vec);
        assert_eq!(emb.dim, 3);
    }

    #[test]
    fn test_embedding_l2_norm_computed() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        let model_hash = [0u8; 32];

        // Non-zero vector should have positive norm
        let vector = vec![0.5, 0.5, 0.5, 0.5];
        let emb = Embedding::new(chunk_id, &vector, model_hash);
        assert!(emb.l2_norm > 0.0, "l2_norm should be positive for non-zero vectors");

        // For a unit vector quantized to 32767, norm should be close to 32767
        let unit_vec = vec![1.0, 0.0];
        let emb2 = Embedding::new(chunk_id, &unit_vec, model_hash);
        assert!((emb2.l2_norm - 32767.0).abs() < 1.0);
    }

    #[test]
    fn test_l2_norm_from_quantized() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        let model_hash = [0u8; 32];
        let vec = vec![100i16, 200, 300];

        let emb = Embedding::from_quantized(chunk_id, vec.clone(), model_hash);

        // Manual calculation: sqrt(100^2 + 200^2 + 300^2) = sqrt(10000 + 40000 + 90000) = sqrt(140000)
        let expected = (140000.0_f64).sqrt() as f32;
        assert!((emb.l2_norm - expected).abs() < 0.01);
    }

    #[test]
    fn test_l2_norm_with_precomputed() {
        let chunk_id = Uuid::from_bytes([0u8; 16]);
        let model_hash = [0u8; 32];
        let vec = vec![100i16, 200, 300];
        let precomputed_norm = 374.17;

        let emb = Embedding::from_quantized_with_norm(chunk_id, vec, model_hash, precomputed_norm);
        assert_eq!(emb.l2_norm, precomputed_norm);
    }
}
