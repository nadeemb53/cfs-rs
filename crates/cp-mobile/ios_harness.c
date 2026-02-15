#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// FFI Declarations (Simplified)
typedef struct CPContext CPContext;

extern CPContext* cp_init(const char* db_path);
extern int cp_sync(CPContext* ctx, const char* relay_url, const char* key_hex);
extern char* cp_query(CPContext* ctx, const char* query);
extern char* cp_get_state_root(CPContext* ctx);
extern void cp_free_string(char* s);
extern void cp_free(CPContext* ctx);
extern const char* cp_version();

int main(int argc, char** argv) {
    printf("--- CP iOS Minimal Harness (V0) ---\n");
    printf("Library Version: %s\n", cp_version());

    const char* db_path = "./mobile_graph.db";
    const char* relay_url = "http://localhost:8080";
    const char* key_hex = "0000000000000000000000000000000000000000000000000000000000000001";

    // 1. Initialize
    printf("Initializing context at %s...\n", db_path);
    CPContext* ctx = cp_init(db_path);
    if (!ctx) {
        fprintf(stderr, "Failed to initialize CP context\n");
        return 1;
    }

    // 2. Pull Sync
    printf("Pulling sync from %s...\n", relay_url);
    int diffs = cp_sync(ctx, relay_url, key_hex);
    if (diffs < 0) {
        fprintf(stderr, "Sync failed with error code: %d\n", diffs);
    } else {
        printf("Sync successful! Applied %d diffs.\n", diffs);
    }

    // 3. Verify State Root
    char* root = cp_get_state_root(ctx);
    if (root) {
        printf("Current State Root: %s\n", root);
        cp_free_string(root);
    }

    // 4. Run Query
    const char* query_text = "troubleshoot";
    printf("Running query: '%s'...\n", query_text);
    char* results_json = cp_query(ctx, query_text);
    if (results_json) {
        printf("Query Results (JSON): %s\n", results_json);
        cp_free_string(results_json);
    }

    // 5. Cleanup
    printf("Cleaning up...\n");
    cp_free(ctx);
    printf("Harness execution complete.\n");

    return 0;
}
