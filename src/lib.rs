pub mod adam;
pub mod dataset_loader;
pub mod dropout;
pub mod embeddings;
pub mod feed_forward;
pub mod layer_norm;
pub mod llm;
pub mod output_projection;
pub mod position_encoding;
pub mod semantic_enhancer;
pub mod self_attention;
pub mod transformer;
pub mod vocab;
// Re-export key structs for easier access
pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use llm::{LLM, Layer};
pub use vocab::Vocab;

// Constants
pub const MAX_SEQ_LEN: usize = 256;  // Increased to accommodate longer Chinese sentences
pub const EMBEDDING_DIM: usize = 512;  // Increased to better represent Chinese characters
pub const HIDDEN_DIM: usize = 1024;  // Increased to handle more complex Chinese language patterns
pub const VOCAB_SIZE: usize = 30000;  // Target vocabulary size for Chinese characters and words
