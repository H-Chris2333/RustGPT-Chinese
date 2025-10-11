# 🦀 RustGPT-Chinese - Chinese-Only LLM

[![Check](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/check.yml/badge.svg)](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/check.yml) [![Test](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/test.yml/badge.svg)](https://github.com/H-Chris233/RustGPT-Chinese/actions/workflows/test.yml)

**[中文文档](README_zh.md) | [中文文档](README_zh.md)**

A complete **Chinese-only Large Language Model implementation in pure Rust** with no external ML frameworks. Built from the ground up using only `ndarray` for matrix operations.

## 🚀 What This Is

This project demonstrates how to build a transformer-based language model from scratch in Rust that is specialized for Chinese language processing, including:
- **Pre-training** on Chinese factual text completion
- **Instruction tuning** for Chinese conversational AI
- **Interactive chat mode** for Chinese language testing
- **Full backpropagation** with gradient clipping
- **Modular architecture** with clean separation of concerns

## ❌ What This Isn't

This is not a production grade Chinese LLM. It is so far away from the larger Chinese models.

This is just a toy project that demonstrates how Chinese LLMs work under the hood.

## 🔍 Key Files to Explore

Start with these two core files to understand the implementation:

- **[`src/main.rs`](src/main.rs)** - Training pipeline, data preparation, and interactive mode
- **[`src/llm.rs`](src/llm.rs)** - Core LLM implementation with forward/backward passes and training logic

## 🏗️ Architecture

The model uses a **transformer-based architecture** with the following components:

```
Input Text → Tokenization → Embeddings → Transformer Blocks → Output Projection → Predictions
```

### Project Structure

```
src/
├── main.rs              # 🎯 Training pipeline and interactive mode
├── llm.rs               # 🧠 Core LLM implementation and training logic
├── lib.rs               # 📚 Library exports and constants
├── transformer.rs       # 🔄 Transformer block (attention + feed-forward)
├── self_attention.rs    # 👀 Multi-head self-attention mechanism
├── feed_forward.rs      # ⚡ Position-wise feed-forward networks
├── embeddings.rs        # 📊 Token embedding layer
├── output_projection.rs # 🎰 Final linear layer for vocabulary predictions
├── vocab.rs            # 📝 Vocabulary management and tokenization
├── layer_norm.rs       # 🧮 Layer normalization
└── adam.rs             # 🏃 Adam optimizer implementation

tests/
├── llm_test.rs         # Tests for core LLM functionality
├── transformer_test.rs # Tests for transformer blocks
├── self_attention_test.rs # Tests for attention mechanisms
├── feed_forward_test.rs # Tests for feed-forward layers
├── embeddings_test.rs  # Tests for embedding layers
├── vocab_test.rs       # Tests for vocabulary handling
├── adam_test.rs        # Tests for optimizer
└── output_projection_test.rs # Tests for output layer
```

## 🧪 What The Model Learns

The implementation includes two training phases specialized for Chinese:

1. **Pre-training**: Learns Chinese world knowledge from Chinese factual statements
   - "太阳从东方升起，在西方落下"
   - "水由于重力而从高处流向低处"
   - "山脉是高大而多岩石的地形"

2. **Instruction Tuning**: Learns Chinese conversational patterns
   - "用户：山脉是如何形成的？助手：山脉通过构造力或火山活动形成..."
   - Handles Chinese greetings, explanations, and follow-up questions

## 🚀 Quick Start

```bash
# Clone and run
git clone https://github.com/H-Chris233/RustGPT-Chinese.git
cd RustGPT-Chinese
cargo run

# The model will:
# 1. Build vocabulary from Chinese training data
# 2. Pre-train on Chinese factual statements (100 epochs)
# 3. Instruction-tune on Chinese conversational data (100 epochs)
# 4. Enter interactive mode for Chinese testing
```

## 🎮 Interactive Mode

After training, test the model interactively in Chinese:

```
Enter prompt: 山脉是如何形成的?
Model output: 山脉通过构造力或火山活动在长时间的地质时期内形成

Enter prompt: 降雨的原因是什么?
Model output: 降雨是由云中的水蒸气凝结成水滴，当水滴变得太重而无法悬浮在空气中时形成的
```

## 🧮 Technical Implementation

### Model Configuration
- **Vocabulary Size**: Dynamic (built from training data)
- **Embedding Dimension**: 128 (defined by `EMBEDDING_DIM` in `src/lib.rs`)
- **Hidden Dimension**: 256 (defined by `HIDDEN_DIM` in `src/lib.rs`)
- **Max Sequence Length**: 80 tokens (defined by `MAX_SEQ_LEN` in `src/lib.rs`)
- **Architecture**: 3 Transformer blocks + embeddings + output projection

### Training Details
- **Optimizer**: Adam with gradient clipping
- **Pre-training LR**: 0.0005 (100 epochs)
- **Instruction Tuning LR**: 0.0001 (100 epochs)
- **Loss Function**: Cross-entropy loss
- **Gradient Clipping**: L2 norm capped at 5.0

### Key Features
- **Custom tokenization** with punctuation handling
- **Greedy decoding** for text generation
- **Gradient clipping** for training stability
- **Modular layer system** with clean interfaces
- **Comprehensive test coverage** for all components

## 🔧 Development

```bash
# Run all tests
cargo test

# Test specific components
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test

# Build optimized version
cargo build --release

# Run with verbose output
cargo test -- --nocapture
```

## 🧠 Learning Resources

This implementation demonstrates key ML concepts for Chinese language models:
- **Transformer architecture** (attention, feed-forward, layer norm)
- **Backpropagation** through neural networks
- **Chinese language model training** (pre-training + fine-tuning)
- **Chinese tokenization** and vocabulary management
- **Gradient-based optimization** with Adam

Perfect for understanding how Chinese LLMs work under the hood!

## 📊 Dependencies

- `ndarray` - N-dimensional arrays for matrix operations
- `rand` + `rand_distr` - Random number generation for initialization

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!

## 🤝 Contributing

Contributions are welcome! This project is perfect for learning and experimentation.

### High Priority Features Needed
- **🏪 Model Persistence** - Save/load trained parameters to disk (currently all in-memory)
- **⚡ Performance optimizations** - SIMD, parallel training, memory efficiency
- **🎯 Better sampling** - Beam search, top-k/top-p, temperature scaling
- **📊 Evaluation metrics** - Perplexity, benchmarks, training visualizations

### Areas for Improvement
- **Advanced architectures** (multi-head attention, positional encoding, RoPE)
- **Training improvements** (different optimizers, learning rate schedules, regularization)
- **Chinese data handling** (larger Chinese datasets, Chinese tokenizer improvements, streaming)
- **Model analysis** (attention visualization, gradient analysis, interpretability)

### Getting Started
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/model-persistence`
3. Make your changes and add tests
4. Run the test suite: `cargo test`
5. Submit a pull request with a clear description

### Code Style
- Follow standard Rust conventions (`cargo fmt`)
- Add comprehensive tests for new features
- Update documentation and README as needed
- Keep the "from scratch" philosophy - avoid heavy ML dependencies
- Focus on Chinese language processing improvements

### Ideas for Contributions
- 🚀 **Beginner**: Model save/load, more Chinese training data, config files
- 🔥 **Intermediate**: Better Chinese tokenization, Chinese-specific optimizations, training checkpoints
- ⚡ **Advanced**: Multi-head attention improvements for Chinese, layer parallelization, custom Chinese optimizations

Questions? Open an issue or start a discussion!

No PyTorch, TensorFlow, or Candle - just pure Rust and linear algebra!
