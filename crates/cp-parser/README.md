# CP Parser

Document parsing for PDF, Markdown, and text files.

## Overview

CP Parser provides:
- File type detection by extension and magic bytes
- Markdown parsing
- Plain text handling
- Chunking with sentence boundary awareness

## Usage

```rust
use cp_parser::{parse_file, Parser};

let content = parse_file("/path/to/document.md")?;
```
