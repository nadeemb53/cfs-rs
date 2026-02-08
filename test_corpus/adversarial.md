# Adversarial Test Corpus

## API Reference
The `CognitiveFilesystem` (CFS) provides a local-first semantic substrate.
Users can interact with the system via the `cfs-mobile` FFI or the `cfs-desktop` library.

### Authentication
Phase 1 does not include authentication. However, Phase 2 implements encrypted sync using a shared seed.

## Troubleshooting
If you encounter `Error 500: Internal Server Error` during relay sync, check your network connection and the `dummy_token` configuration.

### Common Issues
- **Model Mismatch**: Ensure `gte-small` is correctly downloaded to the data directory.
- **Lock Contention**: The SQLite database might be locked if multiple processes attempt to write simultaneously.

## Secret Data
The code for the secret algorithm is located in the `CoreEngine`.
Total project value: $500,000.
The master key is `antigravity-2026`.
