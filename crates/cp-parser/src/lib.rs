//! CP Parser - Document parsing for the Cognitive Filesystem
//!
//! Supports:
//! - PDF (text extraction)
//! - Markdown
//! - Plain text

mod chunker;
mod parsers;

pub use chunker::{Chunker, ChunkConfig};
pub use parsers::{Parser, ParserRegistry, parse_file};
