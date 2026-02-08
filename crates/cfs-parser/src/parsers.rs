//! Document parsers for different file formats

use cfs_core::{CfsError, Result};
use std::path::Path;

/// Trait for document parsers
pub trait Parser: Send + Sync {
    /// Parse a file and return its text content
    fn parse(&self, path: &Path) -> Result<String>;
    
    /// Get supported file extensions
    fn supported_extensions(&self) -> &[&str];
}

/// Registry of available parsers
pub struct ParserRegistry {
    parsers: Vec<Box<dyn Parser>>,
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ParserRegistry {
    /// Create a new registry with default parsers
    pub fn new() -> Self {
        Self {
            parsers: vec![
                Box::new(MarkdownParser),
                Box::new(TextParser),
                Box::new(PdfParser),
            ],
        }
    }

    /// Find a parser for the given file extension
    pub fn find_parser(&self, extension: &str) -> Option<&dyn Parser> {
        for parser in &self.parsers {
            if parser
                .supported_extensions()
                .iter()
                .any(|e| e.eq_ignore_ascii_case(extension))
            {
                return Some(parser.as_ref());
            }
        }
        None
    }
}

/// Parse a file using the appropriate parser
pub fn parse_file(path: &Path) -> Result<String> {
    let registry = ParserRegistry::new();
    
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| CfsError::Parse("No file extension".into()))?;

    let parser = registry
        .find_parser(extension)
        .ok_or_else(|| CfsError::Parse(format!("No parser for extension: {}", extension)))?;

    parser.parse(path)
}

/// Markdown parser using pulldown-cmark
struct MarkdownParser;

impl Parser for MarkdownParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        
        // Convert markdown to plain text
        let parser = pulldown_cmark::Parser::new(&content);
        let mut text = String::new();
        
        for event in parser {
            match event {
                pulldown_cmark::Event::Text(t) 
                | pulldown_cmark::Event::Code(t) => {
                    text.push_str(&t);
                }
                pulldown_cmark::Event::SoftBreak 
                | pulldown_cmark::Event::HardBreak => {
                    text.push('\n');
                }
                pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Paragraph) => {
                    text.push_str("\n\n");
                }
                _ => {}
            }
        }
        
        Ok(normalize_text(&text))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["md", "markdown"]
    }
}

/// Plain text parser
struct TextParser;

impl Parser for TextParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let content = std::fs::read_to_string(path)?;
        Ok(normalize_text(&content))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["txt", "text"]
    }
}

/// PDF parser using pdf-extract
struct PdfParser;

impl Parser for PdfParser {
    fn parse(&self, path: &Path) -> Result<String> {
        let bytes = std::fs::read(path)?;
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| CfsError::Parse(format!("PDF extraction failed: {}", e)))?;
        Ok(normalize_text(&text))
    }

    fn supported_extensions(&self) -> &[&str] {
        &["pdf"]
    }
}

/// Normalize text: trim, collapse whitespace, normalize unicode
fn normalize_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_finds_markdown() {
        let registry = ParserRegistry::new();
        assert!(registry.find_parser("md").is_some());
        assert!(registry.find_parser("markdown").is_some());
    }

    #[test]
    fn test_registry_finds_pdf() {
        let registry = ParserRegistry::new();
        assert!(registry.find_parser("pdf").is_some());
    }

    #[test]
    fn test_unknown_extension() {
        let registry = ParserRegistry::new();
        assert!(registry.find_parser("xyz").is_none());
    }
}
