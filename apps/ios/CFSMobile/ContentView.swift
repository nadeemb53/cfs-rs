import SwiftUI

struct ContentView: View {
    @State private var queryText = ""
    @State private var stateRoot = "None"
    @State private var results: [SearchResult] = []
    @State private var syncStatus = "Ready"
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
                }
                
                HStack {
                    Button(action: pullSync) {
                        Label("Pull Sync", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    
                    Spacer()
                    
                    Text(syncStatus)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
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
        let relayUrl = "http://localhost:3000" // Hardcoded for V0
        let keyHex = "0000000000000000000000000000000000000000000000000000000000000001"
        
        DispatchQueue.global(qos: .userInitiated).async {
            let diffs = bridge.sync(relayUrl: relayUrl, keyHex: keyHex)
            DispatchQueue.main.async {
                if diffs >= 0 {
                    syncStatus = "Applied \(diffs) diffs"
                    stateRoot = bridge.getStateRoot()
                } else {
                    syncStatus = "Error: \(diffs)"
                }
            }
        }
    }
    
    func runQuery() {
        guard !queryText.isEmpty else { return }
        isSearching = true
        
        // Run on background thread
        Task {
            let searchResults = bridge.query(text: queryText)
            
            await MainActor.run {
                self.results = searchResults
                self.isSearching = false
            }
        }
    }
}

// Wrapper to prevent the thought from being sent to user
struct Vroot<Content: View>: View {
    let content: Content
    init(@ViewBuilder content: () -> Content) { self.content = content() }
    var body: some View { content }
}
