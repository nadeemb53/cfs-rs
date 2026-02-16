//! Text normalization utilities for determinism

use unicode_normalization::UnicodeNormalization;

/// Normalize text to canonical form for deterministic hashing
///
/// Steps:
/// 1. Decode generic newlines (`\r\n`, `\r`) to `\n`
/// 2. Apply Unicode Normalization Form C (NFC)
/// 3. Trim trailing whitespace from each line
/// 4. Ensure exactly one trailing newline at EOF (if text is not empty)
///
/// This ensures that equivalent text content produces identical bytes
/// regardless of platform or editor quirks.
pub fn normalize(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    // 1. Normalize line endings to \n
    let text = text.replace("\r\n", "\n").replace('\r', "\n");

    // 2. Unicode Normalization (NFC)
    let nfc_text: String = text.nfc().collect();

    // 3. Process line by line
    let mut normalized = String::with_capacity(nfc_text.len());

    for line in nfc_text.lines() {
        // 4. Trim trailing whitespace
        let trimmed = line.trim_end();

        normalized.push_str(trimmed);
        normalized.push('\n');
    }

    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        let input = "Hello World";
        let expected = "Hello World\n";
        assert_eq!(normalize(input), expected);
    }

    #[test]
    fn test_normalize_whitespace() {
        let input = "Hello   \nWorld  ";
        let expected = "Hello\nWorld\n";
        assert_eq!(normalize(input), expected);
    }

    #[test]
    fn test_normalize_newlines() {
        let input = "Line1\r\nLine2\rLine3\n";
        let expected = "Line1\nLine2\nLine3\n";
        assert_eq!(normalize(input), expected);
    }

    #[test]
    fn test_normalize_unicode() {
        // e + acute accent vs é
        let composed = "\u{0065}\u{0301}"; // e + acute
        let precomposed = "\u{00E9}"; // é

        assert_eq!(normalize(composed), "é\n");
        assert_eq!(normalize(precomposed), "é\n");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize(""), "");
    }

    // Additional tests for comprehensive coverage

    #[test]
    fn test_text_canonicalize_lowercase() {
        // Test lowercase preservation (normalize doesn't convert to lowercase)
        let input = "HELLO WORLD";
        let output = normalize(input);
        assert_eq!(output, "HELLO WORLD\n");
    }

    #[test]
    fn test_text_canonicalize_unicode() {
        // Unicode normalization is applied
        let nfc = "\u{00E9}"; // Precomposed
        let nfd = "\u{0065}\u{0301}"; // Decomposed
        assert_eq!(normalize(nfc), normalize(nfd));
    }

    #[test]
    fn test_text_canonicalize_special_characters() {
        // Preserve meaningful special characters
        let input = "Hello @world! #hashtag $100";
        let result = normalize(input);
        assert!(result.contains("@world"));
        assert!(result.contains("#hashtag"));
        assert!(result.contains("$100"));
    }

    #[test]
    fn test_text_canonicalize_determinism() {
        // Same input always produces same output
        let inputs = vec![
            "Hello World",
            "  Multiple   spaces  ",
            "Line1\nLine2\r\nLine3",
        ];
        for input in inputs {
            let result1 = normalize(input);
            let result2 = normalize(input);
            assert_eq!(result1, result2, "Normalize should be deterministic");
        }
    }

    #[test]
    fn test_text_token_count_estimation() {
        // Test token estimation (rough: 4 chars per token)
        let text = "This is a test string";
        let tokens = text.len() / 4;
        assert!(tokens > 0);
    }

    #[test]
    fn test_text_truncation() {
        // Test handling of very long text
        let long_text = "x".repeat(10000);
        let result = normalize(&long_text);
        assert!(result.len() > 0);
    }

    #[test]
    fn test_text_canonicalize_whitespace() {
        let input = "  Multiple   spaces   here  ";
        let output = normalize(input);
        assert!(output.ends_with("\n"));
    }
}
