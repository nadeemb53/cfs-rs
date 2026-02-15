import Foundation

/// Swift wrapper for the CP Mobile C FFI
class CPBridge {
    private var context: OpaquePointer?
    
    init(dbPath: String) {
        self.context = cp_init(dbPath)
    }
    
    deinit {
        if let context = context {
            cp_free(context)
        }
    }
    
    func sync(relayUrl: String, keyHex: String) -> Int32 {
        guard let context = context else { return -1 }
        return cp_sync(context, relayUrl, keyHex)
    }
    
    func query(text: String) -> [SearchResult] {
        guard let context = context else { return [] }
        guard let cJson = cp_query(context, text) else { return [] }
        defer { cp_free_string(cJson) }
        
        let json = String(cString: cJson)
        let data = json.data(using: .utf8)!
        return (try? JSONDecoder().decode([SearchResult].self, from: data)) ?? []
    }
    
    func getStateRoot() -> String {
        guard let context = context else { return "None" }
        guard let cStr = cp_get_state_root(context) else { return "Error" }
        defer { cp_free_string(cStr) }
        return String(cString: cStr)
    }
    
    func getLastError() -> String {
        guard let cStr = cp_last_error() else { return "Unknown error" }
        defer { cp_free_string(cStr) }
        return String(cString: cStr)
    }
    func getStats() -> String {
        guard let context = context else { return "{}" }
        guard let cStr = cp_stats(context) else { return "{}" }
        defer { cp_free_string(cStr) }
        return String(cString: cStr)
    }

    /// Test if llama.cpp backend can initialize (for debugging)
    func testLlmBackend() -> Int32 {
        return cp_test_llm_backend()
    }

    /// Initialize the LLM with a GGUF model file
    func initLlm(modelPath: String) -> Int32 {
        guard let context = context else { return -1 }
        return cp_init_llm(context, modelPath)
    }

    /// Generate an AI answer using RAG
    func generate(query: String) -> GenerationResult? {
        guard let context = context else { return nil }
        guard let cJson = cp_generate(context, query) else { return nil }
        defer { cp_free_string(cJson) }

        let json = String(cString: cJson)
        let data = json.data(using: .utf8)!
        return try? JSONDecoder().decode(GenerationResult.self, from: data)
    }

    /// Check if the substrate state is valid
    func isStateValid() -> Bool {
        guard let context = context else { return false }
        return cp_is_state_valid(context) == 1
    }

    /// Force a fresh resync by clearing all local data
    /// Use this to recover from verification failures
    func forceResync() -> Int32 {
        guard let context = context else { return -1 }
        return cp_force_resync(context)
    }
}

struct SearchResult: Codable, Identifiable {
    var id: UUID { UUID() }
    let text: String
    let score: Float
    let doc_path: String
}

struct GenerationResult: Codable {
    let answer: String
    let context: String
    let latency_ms: Int
}

// C FFI Prototypes (Must match cp-mobile/src/lib.rs)
// Note: In a real project, these are generated in a bridging header.
@_silgen_name("cp_init")
func cp_init(_ db_path: UnsafePointer<Int8>) -> OpaquePointer?

@_silgen_name("cp_test_llm_backend")
func cp_test_llm_backend() -> Int32

@_silgen_name("cp_init_llm")
func cp_init_llm(_ ctx: OpaquePointer, _ model_path: UnsafePointer<Int8>) -> Int32

@_silgen_name("cp_generate")
func cp_generate(_ ctx: OpaquePointer, _ query: UnsafePointer<Int8>) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cp_sync")
func cp_sync(_ ctx: OpaquePointer, _ relay_url: UnsafePointer<Int8>, _ key_hex: UnsafePointer<Int8>) -> Int32

@_silgen_name("cp_query")
func cp_query(_ ctx: OpaquePointer, _ query: UnsafePointer<Int8>) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cp_get_state_root")
func cp_get_state_root(_ ctx: OpaquePointer) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cp_stats")
func cp_stats(_ ctx: OpaquePointer) -> UnsafeMutablePointer<Int8>?

@_silgen_name("cp_last_error")
func cp_last_error() -> UnsafeMutablePointer<Int8>?

@_silgen_name("cp_free_string")
func cp_free_string(_ s: UnsafeMutablePointer<Int8>)

@_silgen_name("cp_free")
func cp_free(_ ctx: OpaquePointer)

@_silgen_name("cp_is_state_valid")
func cp_is_state_valid(_ ctx: OpaquePointer) -> Int32

@_silgen_name("cp_force_resync")
func cp_force_resync(_ ctx: OpaquePointer) -> Int32
