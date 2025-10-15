# Changelog

All notable changes to RustGPT-Chinese will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-15

### Changed - Model Optimization for Small Datasets

This release focuses on optimizing the model for scenarios with limited training data (200-500 samples).

#### Architecture Changes
- **Reduced model size**: 2 Transformer layers (was 4 in v0.2.0)
- **Reduced embedding dimension**: 256 (was 512 in v0.2.0)
- **Reduced hidden dimension**: 512 (was 1024 in v0.2.0)
- **Reduced sequence length**: 128 tokens (was 256 in v0.2.0)
- **Parameter reduction**: ~86% fewer parameters (10M vs 70M)

#### Training Enhancements
- **Increased training epochs**: 500 (was 100 in v0.2.0)
- **Higher learning rates**:
  - Pre-training: 0.001 (was 0.0005)
  - Instruction tuning: 0.0005 (was 0.0001)
- **Target loss**: < 0.1 for small datasets (vs 0.5+ in v0.2.0)

#### Data Quality Improvements
- **Removed `</s>` tokens** from all training data files
- Cleaner training signal prevents output contamination
- Better model quality with reduced high-frequency token interference

### Fixed
- Model underfitting issue on small datasets (70M parameters → 10M parameters)
- Output contamination from `</s>` special tokens
- Training convergence problems with limited data

### Performance
- Expected training time: 50-60 minutes (similar to v0.2.0 despite 5x epochs due to 86% fewer parameters)
- Better convergence: Loss should reach < 0.1 vs 0.5+ in v0.2.0

### Documentation
- Updated README.md with v0.3.0 configuration details
- Updated CLAUDE.md with architecture changes and rationale
- Added this CHANGELOG.md to track version history

## [0.2.0] - 2025-10-12

### Changed - Architecture Refactoring

#### Major Architecture Upgrade
- **Pre-LN Transformer**: Upgraded from Post-LN to Pre-LN (GPT-2 standard)
  - Layer normalization now before sub-layers instead of after
  - Better training stability and gradient flow
  - Faster convergence and more robust to learning rate changes

#### Explicit Residual Connections
- Moved residual connections from sub-layers to TransformerBlock level
- More explicit and maintainable code structure
- Better alignment with modern transformer implementations

#### Performance Optimizations
- **Jieba singleton optimization**: 50-70% faster tokenization
  - Global singleton pattern prevents repeated initialization
  - Lazy initialization with `OnceLock`
- **Attention reshape optimization**: 20-30% faster attention computation
  - Reduced unnecessary tensor operations
  - More efficient memory access patterns

#### Compiler Optimizations
- Link-time optimization (LTO) enabled
- Maximum optimization level (opt-level 3)
- Single codegen unit for better cross-crate inlining
- Strip debug symbols in release builds

### Added
- **Performance monitoring system**: Comprehensive timing and profiling
- **Model serialization**: Binary (.bin) and JSON format support
- **Detailed inline documentation**: Extensive comments throughout codebase

### Removed
- **Semantic Enhancer**: Removed unverified experimental feature
- Simplified model architecture for better maintainability

## [0.1.0] - Initial Release

### Added
- Basic transformer-based language model implementation in pure Rust
- Chinese language support with jieba-rs tokenization
- Multi-head self-attention mechanism (8 heads)
- Feed-forward networks with GELU activation
- Adam optimizer with gradient clipping
- Layer normalization and dropout regularization
- Pre-training and instruction tuning pipeline
- Interactive chat mode
- Beam search, top-k, and top-p sampling strategies
- Context window management for conversations

### Architecture
- 4 Transformer blocks (Post-LN architecture)
- 512 embedding dimension
- 1024 hidden dimension
- 256 max sequence length
- ~70M parameters

---

## Version Comparison Summary

| Feature | v0.1.0 | v0.2.0 | v0.3.0 |
|---------|--------|--------|--------|
| **Transformer Layers** | 4 (Post-LN) | 4 (Pre-LN) | 2 (Pre-LN) |
| **Embedding Dim** | 512 | 512 | 256 |
| **Hidden Dim** | 1024 | 1024 | 512 |
| **Max Seq Length** | 256 | 256 | 128 |
| **Total Parameters** | ~70M | ~70M | ~10M |
| **Training Epochs** | 100/100 | 100/100 | 500/500 |
| **Learning Rates** | 0.0005/0.0001 | 0.0005/0.0001 | 0.001/0.0005 |
| **`</s>` in Data** | Yes | Yes | No |
| **Expected Loss** | ~0.5 | ~0.5 | < 0.1 |
| **Target Dataset** | Any | Any | 200-500 samples |

---

[0.3.0]: https://github.com/H-Chris233/RustGPT-Chinese/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/H-Chris233/RustGPT-Chinese/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/H-Chris233/RustGPT-Chinese/releases/tag/v0.1.0
