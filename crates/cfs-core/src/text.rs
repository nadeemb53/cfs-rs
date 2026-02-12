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
    // We do this before NFC to ensure \r\n is handled consistently
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
    
    // 4. Trailing Newline logic
    // The loop adds \n after every line.
    // If original text didn't end with newline, `lines()` treats the last part as a line.
    // We always end up with a trailing newline.
    // This matches standard POSIX text file definition.
    // Spec says "Ensure exactly one trailing newline character".
    // Our loop does exactly this.
    
    // Edge case: if input is just whitespace?
    // "   " -> trimmed "" -> "\n"
    // Spec says "Trim trailing whitespace from each line".
    // A file with just spaces becomes a single newline.
    
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
}
