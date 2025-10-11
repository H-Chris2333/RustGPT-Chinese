# 🦀 RustGPT-Chinese - 从零开始构建大语言模型

[![Check](https://github.com/simfg/RustGPT-Chinese/actions/workflows/check.yml/badge.svg)](https://github.com/simfg/RustGPT-Chinese/actions/workflows/check.yml) [![Test](https://github.com/simfg/RustGPT-Chinese/actions/workflows/test.yml/badge.svg)](https://github.com/simfg/RustGPT-Chinese/actions/workflows/test.yml)

这是一个专门用于中文语言处理的**大型语言模型实现**，使用纯 Rust 构建，不依赖任何外部的机器学习框架。完全基于 `ndarray` 实现矩阵运算。

## 🚀 项目简介

本项目展示了如何在 Rust 中从零开始构建专门处理中文的 Transformer 语言模型，包括：

- **中文预训练**：在中文事实知识文本上进行预训练
- **中文指令微调**：针对中文对话场景进行微调
- **中文交互聊天模式**：支持中文交互式对话
- **完整反向传播**：包含梯度裁剪
- **模块化架构**：清晰的关注点分离

## ❌ 项目不是

这不是一个生产级别的大语言模型，距离大型模型还很远。

这只是一个演示项目，展示了这些模型在底层的工作原理。

## 🔍 关键文件

从以下两个核心文件开始了解实现：

- **[`src/main.rs`](src/main.rs)** - 训练流水线、数据准备和交互模式
- **[`src/llm.rs`](src/llm.rs)** - 核心 LLM 实现，包含前向/后向传播和训练逻辑

## 🏗️ 架构

模型使用 **Transformer 基础架构**，包含以下组件：

```
输入文本 → 分词 → 嵌入 → Transformer 块 → 输出投影 → 预测
```

### 项目结构

```
src/
├── main.rs              # 🎯 训练流水线和交互模式
├── llm.rs               # 🧠 核心 LLM 实现和训练逻辑
├── lib.rs               # 📚 库导出和常量
├── transformer.rs       # 🔄 Transformer 块（注意力 + 前馈）
├── self_attention.rs    # 👀 多头自注意力机制
├── feed_forward.rs      # ⚡ 位置前馈网络
├── embeddings.rs        # 📊 词嵌入层
├── output_projection.rs # 🎰 最终线性层，用于词汇预测
├── vocab.rs            # 📝 词汇管理与分词
├── layer_norm.rs       # 🧮 层归一化
└── adam.rs             # 🏃 Adam 优化器实现

tests/
├── llm_test.rs         # LLM 功能测试
├── transformer_test.rs # Transformer 块测试
├── self_attention_test.rs # 注意力机制测试
├── feed_forward_test.rs # 前馈层测试
├── embeddings_test.rs  # 嵌入层测试
├── vocab_test.rs       # 词汇处理测试
├── adam_test.rs        # 优化器测试
└── output_projection_test.rs # 输出层测试
```

## 🧪 模型学习内容

实现包括两个专门针对中文的训练阶段：

1. **中文预训练**：从中文事实陈述中学习中文世界知识
   - "太阳从东方升起，在西方落下"
   - "水由于重力而从高处流向低处"
   - "山脉是高大而多岩石的地形"
   
2. **中文指令微调**：学习中文对话模式
   - "用户：山脉是如何形成的？助手：山脉通过构造力或火山活动形成..."
   - 处理中文问候、解释和后续问题

## 🚀 快速开始

```bash
# 克隆并运行
git clone https://github.com/simfg/RustGPT-Chinese.git
cd RustGPT-Chinese
cargo run

# 模型将：
# 1. 从中文训练数据构建词汇表
# 2. 在中文事实陈述上进行预训练（100 轮）
# 3. 在中文对话数据上进行指令微调（100 轮）
# 4. 进入中文交互模式进行测试
```

## 🎮 交互模式

训练完成后，可以交互式地测试中文模型：

```
输入提示：山脉是如何形成的？
模型输出：山脉通过构造力或火山活动在长时间的地质时期内形成

输入提示：降雨的原因是什么？
模型输出：降雨是由云中的水蒸气凝结成水滴，当水滴变得太重而无法悬浮在空气中时形成的
```

## 🧮 技术实现

### 模型配置
- **词汇表大小**：动态（基于训练数据构建）
- **嵌入维度**：128（由 `src/lib.rs` 中的 `EMBEDDING_DIM` 定义）
- **隐藏维度**：256（由 `src/lib.rs` 中的 `HIDDEN_DIM` 定义）
- **最大序列长度**：80 个标记（由 `src/lib.rs` 中的 `MAX_SEQ_LEN` 定义）
- **架构**：3 个 Transformer 块 + 嵌入 + 输出投影

### 训练细节
- **优化器**：Adam + 梯度裁剪
- **预训练学习率**：0.0005（100 轮）
- **指令微调学习率**：0.0001（100 轮）
- **损失函数**：交叉熵损失
- **梯度裁剪**：L2 范数限制为 5.0

### 关键特性
- **自定义分词**：支持标点符号处理
- **贪婪解码**：用于文本生成
- **梯度裁剪**：训练稳定性
- **模块化层系统**：清晰的接口
- **全面的测试覆盖**：所有组件

## 🔧 开发

```bash
# 运行所有测试
cargo test

# 测试特定组件
cargo test --test llm_test
cargo test --test transformer_test
cargo test --test self_attention_test
cargo test --test feed_forward_test

# 构建优化版本
cargo build --release

# 运行详细输出
cargo test -- --nocapture
```

## 🧠 学习资源

此实现展示了中文大语言模型的关键机器学习概念：
- **Transformer 架构**（注意力、前馈、层归一化）
- **通过神经网络的反向传播**
- **中文语言模型训练**（预训练 + 微调）
- **中文分词和词汇管理**
- **基于梯度的 Adam 优化**

是了解现代 LLM 在底层如何工作的完美选择！

## 📊 依赖项

- `ndarray` - 用于矩阵运算的 N 维数组
- `rand` + `rand_distr` - 随机数生成
- `serde` + `serde_json` - 序列化和 JSON 处理
- `csv` - CSV 文件处理
- `bincode` - 二进制序列化

没有 PyTorch、TensorFlow 或 Candle - 只有纯 Rust 和线性代数！

## 🤝 贡献

欢迎贡献！这个项目非常适合学习和实验。

### 高优先级功能需求
- **🏪 模型持久化** - 将训练参数保存到磁盘（目前全部在内存中）
- **⚡ 性能优化** - SIMD、并行训练、内存效率
- **🎯 更好的采样** - 波束搜索、top-k/top-p、温度缩放
- **📊 评估指标** - 困惑度、基准测试、训练可视化

### 改进领域
- **高级架构**（多头注意力、位置编码、RoPE）
- **训练改进**（不同优化器、学习率调度、正则化）
- **中文数据处理**（更大的中文数据集、中文分词器改进、流式处理）
- **模型分析**（注意力可视化、梯度分析、可解释性）

### 入门
1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/model-persistence`
3. 进行更改并添加测试
4. 运行测试套件：`cargo test`
5. 提交 PR 并提供清晰的描述

### 代码风格
- 遵循标准 Rust 约定（`cargo fmt`）
- 为新功能添加全面测试
- 根据需要更新文档和 README
- 保持"从零开始"的哲学 - 避免重量级 ML 依赖

### 贡献想法
- 🚀 **初学者**：模型保存/加载、更多中文训练数据、配置文件
- 🔥 **中级**：中文分词优化、位置编码、训练检查点
- ⚡ **高级**：多头注意力改进用于中文、层并行化、自定义中文优化

有疑问？开一个 issue 或开始讨论！

没有 PyTorch、TensorFlow 或 Candle - 只有纯 Rust 和线性代数！