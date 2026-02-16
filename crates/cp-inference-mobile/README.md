# CP Inference Mobile

Local LLM inference for CP mobile (llama.cpp based).

## Overview

CP Inference Mobile provides:
- llama.cpp bindings for mobile LLM inference
- On-device generation
- Mobile-optimized memory management

## Usage

```rust
use cp_inference_mobile::LlamaRunner;

let mut runner = LlamaRunner::new(model_path)?;
let output = runner.generate("Prompt", max_tokens)?;
```
