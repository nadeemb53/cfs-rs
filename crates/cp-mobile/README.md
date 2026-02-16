# CP Mobile

Mobile-optimized CP core with C FFI for Canon Protocol (CP).

## Overview

CP Mobile provides:
- C-compatible FFI for embedding in mobile apps
- iOS and Android support
- Minimal dependencies for small binary size

## Usage

```rust
// Mobile apps use the C FFI
#[no_mangle]
pub extern "C" fn cp_init() -> *mut CPContext {
    // Initialize CP core
}
```
