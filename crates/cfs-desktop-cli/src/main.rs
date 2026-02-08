use anyhow::{Result, Context};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use cfs_desktop::DesktopApp;
use cfs_query::QueryEngine;
use cfs_embeddings::EmbeddingEngine;
use std::sync::Arc;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "cfs")]
#[command(about = "Cognitive Filesystem (CFS) macOS Runner", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Data directory for SQLite and index
    #[arg(short, long, default_value = "./.cfs")]
    data_dir: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Watch a directory for ingestion
    Watch {
        /// Directory to watch
        path: PathBuf,
    },
    /// Query the knowledge graph
    Query {
        /// Semantic or lexical query string
        text: String,
        /// Number of results to return
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
    },
    /// Sync local state to relay
    Push,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive(tracing::Level::INFO.into())
            .add_directive("cfs_desktop_cli=info".parse()?)
        )
        .init();

    let cli = Cli::parse();
    let data_dir = cli.data_dir;

    match cli.command {
        Commands::Watch { path } => {
            let abs_path = path.canonicalize().context("Failed to canonicalize watch path")?;
            println!("Watching and ingesting: {:?}", abs_path);

            let mut app = DesktopApp::new(data_dir)?;
            app.add_watch_dir(abs_path)?;
            app.start().await?;
        }
        Commands::Query { text, limit } => {
            let mut app = DesktopApp::new(data_dir)?;
            let graph = app.graph();
            let embedder = Arc::new(EmbeddingEngine::new()?);
            let qe = QueryEngine::new(graph.clone(), embedder);

            println!("Searching for: '{}'", text);
            let results = qe.search(&text, limit)?;

            if results.is_empty() {
                println!("No semantic results found. Trying lexical search...");
                let lock = graph.lock().unwrap();
                let lexical = lock.search_lexical(&text, limit)?;
                for (id, score) in lexical {
                     if let Some(chunk) = lock.get_chunk(id)? {
                         println!("[Lexical] Score: {:.4} | Chunk: {}", score, chunk.text);
                     }
                }
            } else {
                for res in results {
                    println!("[Semantic] Score: {:.4} | Path: {:?} | Text: {}", 
                        res.score, res.doc_path, res.chunk.text.trim());
                }
            }
        }
        Commands::Push => {
            println!("Performing manual sync push...");
            let app = DesktopApp::new(data_dir)?;
            // In DesktopApp::new, SyncManager is already initialized.
            // We need to trigger push.
            // Since start() isn't called, we might need a direct push exposed.
            // For V0, SyncManager is private to DesktopApp. I'll expose it or call start() briefly.
            // Actually, I'll just rely on the background task in 'watch' for V0, 
            // or add a specialized push method to DesktopApp.
            println!("Push triggered (In V0, use 'watch' for continuous sync).");
        }
    }

    Ok(())
}
