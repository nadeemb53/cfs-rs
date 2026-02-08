use cfs_core::{Chunk, ScoredChunk, ContextAssembler};
use uuid::Uuid;

#[test]
fn test_context_assembly_determinism() {
    let doc_id = Uuid::new_v4();
    let c1 = Chunk::new(doc_id, "Chunk 1".to_string(), 0, 0);
    let c2 = Chunk::new(doc_id, "Chunk 2".to_string(), 10, 1);
    
    let sc1 = ScoredChunk { chunk: c1.clone(), score: 0.9 };
    let sc2 = ScoredChunk { chunk: c2.clone(), score: 0.8 };
    
    let assembler = ContextAssembler::new(1000);
    
    // Order 1
    let context_a = assembler.assemble(vec![sc1.clone(), sc2.clone()]);
    
    // Order 2 (reversed input)
    let context_b = assembler.assemble(vec![sc2.clone(), sc1.clone()]);
    
    assert_eq!(context_a, context_b, "Context assembly must be deterministic regardless of input order");
    assert!(context_a.contains("Chunk 1"));
    assert!(context_a.contains("Chunk 2"));
}

#[test]
fn test_context_token_budget() {
    let doc_id = Uuid::new_v4();
    let c1 = Chunk::new(doc_id, "A".repeat(400), 0, 0); // ~100 tokens
    let c2 = Chunk::new(doc_id, "B".repeat(400), 10, 1); // ~100 tokens
    
    let sc1 = ScoredChunk { chunk: c1, score: 0.9 };
    let sc2 = ScoredChunk { chunk: c2, score: 0.8 };
    
    let assembler = ContextAssembler::new(150); // Only enough for ~1 chunk (approx_tokens uses /4)
    // Actually approx_tokens is text.len() / 4. 400/4 = 100.
    // 150 budget should allow one chunk.
    
    let context = assembler.assemble(vec![sc1, sc2]);
    assert!(context.contains("Chunk 0"));
    assert!(!context.contains("Chunk 1"), "Must respect token budget");
}
