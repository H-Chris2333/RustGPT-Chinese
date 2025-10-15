# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustGPT-Chinese is a Chinese-specialized transformer-based language model built from scratch in pure Rust using only `ndarray` for matrix operations. It's an educational project demonstrating how LLMs work at the implementation level, not a production-grade system.

## Development Commands

### Building and Running
```bash
# Run the main training pipeline
cargo run

# Build optimized release version
cargo build --release

# Run with verbose output
cargo test -- --nocapture
```

### Testing
```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test
cargo test --test chinese_tests
cargo test --test vocab_test

# Run individual component tests
cargo test --test feed_forward_test
cargo test --test embeddings_test
cargo test --test output_projection_test
cargo test --test adam_test
cargo test --test dataset_loader_test
cargo test --test position_encoding_test
```

### Code Quality
```bash
# Format code
cargo fmt

# Run linter
cargo clippy
```

## Architecture Overview

### Core Design Philosophy

This is a **pure Rust, from-scratch implementation** with no PyTorch, TensorFlow, or Candle dependencies. All neural network operations are implemented manually using `ndarray` for matrix operations and `jieba-rs` for Chinese tokenization.

### Data Flow

```
Input Text → Jieba Tokenization → Token IDs → Embeddings (256d)
    → 2x Transformer Blocks (attention + FFN + dropout + layer norm)
    → Output Projection → Softmax → Token Predictions
```

### Key Architecture Components

**Network Stack (in order):**
1. **Embeddings Layer** (`embeddings.rs`) - Token embedding optimized for Chinese
2. **2x Transformer Blocks** (`transformer.rs`) - Each contains:
   - Multi-head self-attention (8 heads, `self_attention.rs`)
   - Feed-forward network (`feed_forward.rs`)
   - 2x Dropout layers (10% rate, `dropout.rs`)
   - 2x Layer normalization (`layer_norm.rs`)
3. **Output Projection** (`output_projection.rs`) - Maps to vocabulary size

**Supporting Systems:**
- **Vocabulary Management** (`vocab.rs`) - Handles Chinese tokenization with jieba-rs, idiom detection, special tokens
- **Position Encoding** (`position_encoding.rs`) - Positional information for sequence understanding
- **Semantic Enhancer** (`semantic_enhancer.rs`) - Chinese-specific relationship enhancement
- **Adam Optimizer** (`adam.rs`) - Gradient-based optimization with momentum
- **Dataset Loader** (`dataset_loader.rs`) - Loads pre-training and chat training data from JSON

### Model Configuration (lib.rs) - v0.3.0

```rust
MAX_SEQ_LEN: 128        // Optimized for small datasets (was 256)
EMBEDDING_DIM: 256      // Reduced for better convergence on limited data (was 512)
HIDDEN_DIM: 512         // Reduced for better convergence on limited data (was 1024)
VOCAB_SIZE: 30000       // Target vocab size (dynamically built from data)
```

**Why these changes in v0.3.0?**
- Smaller model = fewer parameters = better fit for 200-500 training samples
- Reduces risk of severe underfitting when training data is limited
- Parameter count reduced from ~70M to ~10M (86% reduction)

### Training Pipeline (main.rs) - v0.3.0

The training process has two phases:

1. **Vocabulary Building**: Processes both pre-training and chat training data using jieba-rs to extract all unique tokens (Chinese words, idioms, punctuation, special tokens)

2. **Pre-training** (500 epochs, LR=0.001):
   - Loads data from `data/pretraining_data.json`
   - Learns Chinese world knowledge and factual statements
   - Higher learning rate and more epochs for small dataset optimization
   - Uses learning rate decay (0.95 per 10 steps)

3. **Instruction Tuning** (500 epochs, LR=0.0005):
   - Loads data from `data/chat_training_data.json`
   - Learns conversational Chinese patterns
   - Higher learning rate and more epochs for small dataset optimization
   - Uses learning rate decay

4. **Interactive Mode**:
   - Beam search decoding (width=3, max_length=20)
   - Context window management for multi-turn conversations
   - Chinese text post-processing to remove extra spaces

**Training Data Changes (v0.3.0):**
- Removed `</s>` tokens from all training data to prevent output contamination
- Cleaner training signal = better model quality

### Chinese Language Handling

**Tokenization Strategy** (vocab.rs, llm.rs):
- Detects Chinese characters (Unicode range 0x4E00-0x9FFF)
- Uses jieba-rs for Chinese word segmentation
- Falls back to whitespace tokenization for non-Chinese text
- Handles Chinese punctuation as separate tokens
- Extracts 4-character Chinese idioms (成语) with pattern matching

**Special Processing:**
- Idiom detection via regex patterns and dictionary lookup (`data/chinese_idioms.json`)
- Phrase extraction for multi-character meaningful phrases
- Post-processing removes extra spaces between Chinese characters

### Training Mechanics (llm.rs)

**Forward Pass:**
- Teacher forcing: input is `tokens[:-1]`, target is `tokens[1:]`
- Each token position predicts the next token
- Supports multiple sampling strategies:
  - Greedy decoding (takes highest probability)
  - Top-k sampling (samples from k most probable tokens)
  - Top-p (nucleus) sampling (samples from smallest set with cumulative prob > p)
  - Beam search (maintains multiple candidate sequences)
  - Temperature scaling for output diversity

**Backward Pass:**
- Cross-entropy loss with numerical stability (`max(1e-15)` clipping)
- Softmax + cross-entropy gradient: `softmax_probs - one_hot(target)`
- Gradient clipping (L2 norm max 5.0) for stability
- Gradients propagated backwards through all layers
- Each layer updates its own parameters with Adam optimizer

**Context Management:**
- Context window maintains conversation history (up to MAX_SEQ_LEN tokens, 128 in v0.3.0)
- Oldest tokens removed when exceeding max length
- Context cleared on `</s>` token detection (though `</s>` removed from training data in v0.3.0)

### Layer Interface (llm.rs)

All layers implement the `Layer` trait:
```rust
trait Layer {
    fn layer_type(&self) -> &str;                          // Returns layer name
    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;  // Forward pass
    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;  // Backward pass
    fn parameters(&self) -> usize;                         // Parameter count
    fn set_training_mode(&mut self, training: bool);       // Toggle dropout
}
```

### Transformer Block Details (transformer.rs)

Each block applies this sequence (Pre-LN architecture):
1. LayerNorm → Self-attention → Dropout (10%) → Residual connection
2. LayerNorm → Feed-forward → Dropout (10%) → Residual connection

Note: v0.2.0 upgraded to Pre-LN architecture with explicit residual connections for better training stability.

### Special Tokens

Defined in `vocab.rs`:
- `<|pad|>` (ID: 0) - Padding
- `<|unk|>` (ID: 1) - Unknown tokens
- `<|bos|>` (ID: 2) - Beginning of sequence
- `</s>` (ID: 3) - End of sequence (note: removed from training data in v0.3.0 but still defined in vocab)
- `<|sep|>` (ID: 4) - Separator
- `<|cls|>` (ID: 5) - Classification
- `<|mask|>` (ID: 6) - Masked token
- `<|bos|>` (ID: 2) - Beginning of sequence
- `</s>` (ID: 3) - End of sequence (used to trigger context clearing)
- `<|sep|>` (ID: 4) - Separator
- `<|cls|>` (ID: 5) - Classification
- `<|mask|>` (ID: 6) - Masked token

## Common Development Patterns

### Adding a New Layer

1. Create the layer struct with parameters and cache for forward pass values
2. Implement the `Layer` trait with forward/backward/parameters/set_training_mode
3. In forward: cache inputs needed for backward pass
4. In backward: compute gradients w.r.t. inputs and update parameters
5. Add the layer to the network stack in `main.rs` or `llm.rs`
6. Create a test file in `tests/` following existing patterns

### Modifying Training Data

Training data is loaded from JSON files in `data/`:
- `data/pretraining_data.json` - Array of Chinese factual statements (no `</s>` tokens in v0.3.0)
- `data/chat_training_data.json` - Array of conversational exchanges (no `</s>` tokens in v0.3.0)
- `data/chinese_idioms.json` - Array of 4-character Chinese idioms

Format: Simple JSON arrays of strings.

**v0.3.0 Note**: `</s>` tokens were removed from training data to prevent output contamination and improve model quality.

### Working with Chinese Text

When processing Chinese text:
- Always check for Chinese characters using: `(char as u32) >= 0x4E00 && (char as u32) <= 0x9FFF`
- Use jieba-rs tokenizer for segmentation: `jieba.cut(text, false)`
- Be aware that tokenization happens in both `vocab.rs` and `llm.rs`
- Post-processing removes spaces between Chinese characters for fluency

### Debugging Model Output

Key files to examine:
- `src/llm.rs:251` - Training loop with loss printing
- `src/llm.rs:95` - Sampling methods (temperature, top-k, top-p)
- `src/llm.rs:122` - Beam search implementation
- `src/llm.rs:742` - Chinese text post-processing
- `src/main.rs:124` - Interactive mode with beam search

## Testing Strategy

Tests are organized by component in the `tests/` directory. Each test file corresponds to a source module. Key test patterns:

- **Forward pass tests**: Verify output shapes and basic functionality
- **Backward pass tests**: Check gradient computation (often by verifying parameters change)
- **Chinese-specific tests**: Validate tokenization, idiom detection, and text processing
- **Integration tests**: Test full training pipeline components together

## Known Limitations

- Model persistence is supported via binary (.bin) and JSON formats (see `model_serialization.rs`)
- Small model size (v0.3.0: 10M parameters) limits generalization beyond training data
- Training data limited to ~250 samples - more data needed for production use
- No attention masking for autoregressive generation (relies on teacher forcing)
- No batching support (processes one sequence at a time)
- Training data is loaded from JSON files (expandable)
- No learning rate warmup (uses exponential decay only)
- No gradient accumulation
- No residual connections in transformer blocks (reduces training stability)
- Limited vocabulary size (dynamically built from training data only)
- No attention masking for autoregressive generation
- No batching support (processes one sequence at a time)
- Training data is hardcoded in JSON files
- No learning rate warmup
- No gradient accumulation

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- `check.yml` - Runs `cargo clippy` and `cargo fmt --check`
- `test.yml` - Runs full test suite with `cargo test`

## Project Philosophy

This project prioritizes **educational clarity over production performance**. Code is intentionally verbose and explicit to demonstrate how transformers work at the implementation level. Avoid adding heavy ML framework dependencies - keep the "from scratch" philosophy intact.
