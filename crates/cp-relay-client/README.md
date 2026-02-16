# CP Relay Client

HTTP client for CP relay server.

## Overview

CP Relay Client provides:
- Client for communicating with CP relay servers
- Push/pull operations
- Connection pooling

## Usage

```rust
use cp_relay_client::RelayClient;

let client = RelayClient::new("http://localhost:8080")?;
client.push(&documents)?;
```
