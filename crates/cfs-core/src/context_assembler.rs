use crate::Chunk;
use std::collections::{HashMap, HashSet};

/// Search result with its score
#[derive(Clone)]
pub struct ScoredChunk {
    pub chunk: Chunk,
    pub score: f32,
}

/// Assembler for deterministic context construction
pub struct ContextAssembler {
    token_budget: usize,
}

impl ContextAssembler {
    pub fn new(token_budget: usize) -> Self {
        Self { token_budget }
    }

    /// Assemble chunks into a deterministic byte-identical context string
    pub fn assemble(&self, chunks: Vec<ScoredChunk>) -> String {
        // 1. Deduplicate by chunk_id
        let mut unique_chunks = HashMap::new();
        for sc in chunks {
            unique_chunks.entry(sc.chunk.id).or_insert(sc);
        }

        let mut chunks: Vec<ScoredChunk> = unique_chunks.into_values().collect();

        // 2. Sort by: Score (Desc), Section (Asc - Seq), Offset (Asc)
        // Note: Using stable sort to ensure identical inputs produce identical outputs
        chunks.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap() // Score Desc
                .then_with(|| a.chunk.seq.cmp(&b.chunk.seq)) // Seq Asc
                .then_with(|| a.chunk.offset.cmp(&b.chunk.offset)) // Offset Asc
        });

        // 3. Pack greedily until token budget
        let mut total_tokens = 0;
        let mut context = String::new();
        let mut handled_docs = HashSet::new();

        for sc in chunks {
            let tokens = sc.chunk.approx_tokens();
            if total_tokens + tokens > self.token_budget {
                break;
            }

            if handled_docs.insert(sc.chunk.doc_id) {
                context.push_str(&format!("\n--- Document: {} ---\n", sc.chunk.doc_id));
            }

            context.push_str(&format!("(Chunk {}): {}\n", sc.chunk.seq, sc.chunk.text));
            total_tokens += tokens;
        }

        context.trim().to_string()
    }
}
