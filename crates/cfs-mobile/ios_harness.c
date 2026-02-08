#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// FFI Declarations (Simplified)
typedef struct CfsContext CfsContext;

extern CfsContext* cfs_init(const char* db_path);
extern int cfs_sync(CfsContext* ctx, const char* relay_url, const char* key_hex);
extern char* cfs_query(CfsContext* ctx, const char* query);
extern char* cfs_get_state_root(CfsContext* ctx);
extern void cfs_free_string(char* s);
extern void cfs_free(CfsContext* ctx);
extern const char* cfs_version();

int main(int argc, char** argv) {
    printf("--- CFS iOS Minimal Harness (V0) ---\n");
    printf("Library Version: %s\n", cfs_version());

    const char* db_path = "./mobile_graph.db";
    const char* relay_url = "http://localhost:3000";
    const char* key_hex = "0000000000000000000000000000000000000000000000000000000000000001";

    // 1. Initialize
    printf("Initializing context at %s...\n", db_path);
    CfsContext* ctx = cfs_init(db_path);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CFS context\n");
        return 1;
    }

    // 2. Pull Sync
    printf("Pulling sync from %s...\n", relay_url);
    int diffs = cfs_sync(ctx, relay_url, key_hex);
    if (diffs < 0) {
        fprintf(stderr, "Sync failed with error code: %d\n", diffs);
    } else {
        printf("Sync successful! Applied %d diffs.\n", diffs);
    }

    // 3. Verify State Root
    char* root = cfs_get_state_root(ctx);
    if (root) {
        printf("Current State Root: %s\n", root);
        cfs_free_string(root);
    }

    // 4. Run Query
    const char* query_text = "troubleshoot";
    printf("Running query: '%s'...\n", query_text);
    char* results_json = cfs_query(ctx, query_text);
    if (results_json) {
        printf("Query Results (JSON): %s\n", results_json);
        cfs_free_string(results_json);
    }

    // 5. Cleanup
    printf("Cleaning up...\n");
    cfs_free(ctx);
    printf("Harness execution complete.\n");

    return 0;
}
