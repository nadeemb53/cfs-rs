import SwiftUI

struct ContentView: View {
    @State private var relayUrl = "http://127.0.0.1:8080"
    @State private var queryText = ""
    @State private var stateRoot = "None"
    @State private var results: [SearchResult] = []
    @State private var syncStatus = "Ready"
    @State private var stats = "Docs: 0, Chunks: 0"
    @State private var isSearching = false
    
    // In V0, we use a single instance of the bridge
    private let bridge: CfsBridge
    
    init() {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let dbPath = paths[0].appendingPathComponent("mobile_graph.db").path
        self.bridge = CfsBridge(dbPath: dbPath)
    }
    
    var body: some View {
        Vroot {
            VStack(alignment: .leading, spacing: 20) {
                Text("CFS iOS (V0)")
                    .font(.largeTitle)
                    .bold()
                
                Group {
                    Text("State Root")
                        .font(.headline)
                    Text(stateRoot)
                        .font(.caption)
                        .monospaced()
                        .padding(8)
                        .background(Color.secondary.opacity(0.1))
                        .cornerRadius(4)
                    
                    Text(stats)
                        .font(.caption2)
                        .foregroundColor(.blue)
                }
                
                HStack {
                    VStack(alignment: .leading) {
                        Text("Relay URL")
                            .font(.caption)
                        TextField("http://...", text: $relayUrl)
                            .textFieldStyle(.roundedBorder)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.none)
                    }
                    
                    Button(action: {
                        pullSync()
                    }) {
                        Label("Pull", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                }
                
                Text(syncStatus)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .lineLimit(2)
                
                Divider()
                
                VStack(alignment: .leading) {
                    TextField("Search chunks...", text: $queryText)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.none)
                    
                    HStack {
                        Button("Run Query") {
                            runQuery()
                        }
                        .buttonStyle(.bordered)
                        .disabled(isSearching || queryText.isEmpty)
                        
                        if isSearching {
                            ProgressView()
                                .padding(.leading, 8)
                        }
                    }
                }
                
                List(results) { res in
                    VStack(alignment: .leading) {
                        Text("Score: \(String(format: "%.4f", res.score))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(res.text)
                            .font(.body)
                        Text(res.doc_path)
                            .font(.caption2)
                            .foregroundColor(.blue)
                    }
                }
                .listStyle(.plain)
            }
            .padding()
        }
    }
    
    func pullSync() {
        syncStatus = "Syncing..."
        let keyHex = "0101010101010101010101010101010101010101010101010101010101010101"
        
        DispatchQueue.global(qos: .userInitiated).async {
            let diffs = bridge.sync(relayUrl: relayUrl, keyHex: keyHex)
            DispatchQueue.main.async {
                if diffs >= 0 {
                    syncStatus = "Applied \(diffs) diffs"
                    stateRoot = bridge.getStateRoot()
                    updateStats()
                } else {
                    let detail = bridge.getLastError()
                    syncStatus = "Error: \(detail)"
                    if detail.contains("connection refused") || detail.contains("localhost") {
                        syncStatus += " (Tip: Use Mac's IP if on real device)"
                    }
                }
            }
        }
    }
    
    func runQuery() {
        guard !queryText.isEmpty else { return }
        syncStatus = "Searching..."
        isSearching = true
        
        // Run on background thread
        Task {
            let searchResults = bridge.query(text: queryText)
            let error = bridge.getLastError()
            
            await MainActor.run {
                self.results = searchResults
                self.isSearching = false
                if searchResults.isEmpty && !error.isEmpty && !error.contains("Mutex") {
                    syncStatus = "Search Error: \(error)"
                } else if searchResults.isEmpty {
                    syncStatus = "No results found"
                } else {
                    syncStatus = "Found \(searchResults.count) results"
                }
            }
        }
    }

    func updateStats() {
        let json = bridge.getStats()
        // Simple parse for V0
        if let data = json.data(using: .utf8),
           let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Int] {
            let docs = dict["documents"] ?? 0
            let chunks = dict["chunks"] ?? 0
            let embs = dict["embeddings"] ?? 0
            stats = "Documents: \(docs), Chunks: \(chunks), Embeddings: \(embs)"
        }
    }
}

// Wrapper to prevent the thought from being sent to user
struct Vroot<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View { content }
}
