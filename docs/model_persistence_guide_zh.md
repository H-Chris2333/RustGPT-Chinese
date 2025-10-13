# 模型持久化使用指南

本文档说明如何使用 RustGPT-Chinese 的模型持久化功能。

## 📚 功能概述

模型持久化功能支持两种格式:

### 1. 二进制格式 (.bin) - 推荐用于生产

**优点:**
- 文件小、加载速度快
- 保存完整的优化器状态(Adam动量)
- 支持断点续训

**适用场景:**
- 日常训练checkpoint保存
- 长期训练中间状态保存
- 生产环境模型部署

### 2. JSON 格式 (.json) - 推荐用于调试

**优点:**
- 人类可读,方便检查权重
- 跨语言兼容,可用Python读取
- 保存完整的优化器状态
- 方便手动编辑和分析

**适用场景:**
- 调试训练过程
- 与其他框架/语言共享权重
- 研究和分析模型参数

## 🚀 快速开始

### 1. 基本使用

```rust
use llm::{LLM, save_model_binary, load_model_binary};

// 创建或训练模型
let mut llm = LLM::default();

// 训练代码...
// llm.train(...);

// 保存模型到二进制文件
save_model_binary(&llm, "checkpoints/model_epoch_100.bin")?;

// 加载模型
let loaded_llm = load_model_binary("checkpoints/model_epoch_100.bin")?;
```

### 2. 使用 JSON 格式

```rust
use llm::{save_model_json, load_model_json};

// 保存为JSON(方便调试)
save_model_json(&llm, "exports/model_weights.json")?;

// 从JSON加载
let loaded_llm = load_model_json("exports/model_weights.json")?;
```

### 3. 自动格式识别

```rust
use llm::load_model_auto;

// 根据文件扩展名自动选择格式
let llm = load_model_auto("checkpoints/model.bin")?;  // 二进制
let llm = load_model_auto("exports/model.json")?;     // JSON
```

## 💡 完整训练示例

```rust
use llm::{LLM, Vocab, save_model_binary, load_model_binary};
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 准备训练数据
    let training_texts = vec![
        "中国是一个历史悠久的国家".to_string(),
        "人工智能技术正在快速发展".to_string(),
        // ... 更多训练数据
    ];

    // 2. 构建词汇表
    let mut vocab_set = HashSet::new();
    Vocab::process_text_for_vocab(&training_texts, &mut vocab_set);
    let vocab_words: Vec<String> = vocab_set.into_iter().collect();
    let vocab = Vocab::new(vocab_words.iter().map(|s| s.as_str()).collect());

    // 3. 创建模型
    let mut llm = LLM::new(vocab, /* network layers */);

    // 4. 训练模型
    let epochs = 100;
    for epoch in 0..epochs {
        let training_refs: Vec<&str> = training_texts.iter().map(|s| s.as_str()).collect();
        llm.train(training_refs, 1, 0.001);

        // 每10个epoch保存一次checkpoint
        if (epoch + 1) % 10 == 0 {
            let checkpoint_path = format!("checkpoints/model_epoch_{}.bin", epoch + 1);
            save_model_binary(&llm, &checkpoint_path)?;
            println!("✓ 已保存checkpoint: {}", checkpoint_path);
        }
    }

    // 5. 保存最终模型
    save_model_binary(&llm, "model_final.bin")?;
    println!("✓ 训练完成,模型已保存!");

    Ok(())
}
```

## 🔄 断点续训示例

```rust
use llm::{load_model_binary, save_model_binary};

fn continue_training() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 从checkpoint加载模型
    println!("📂 加载checkpoint...");
    let mut llm = load_model_binary("checkpoints/model_epoch_50.bin")?;

    // 2. 继续训练
    println!("🔄 继续训练...");
    let training_data = vec!["新的训练数据1", "新的训练数据2"];
    llm.train(training_data, 50, 0.0005);  // 继续训练50个epoch

    // 3. 保存新的checkpoint
    save_model_binary(&llm, "checkpoints/model_epoch_100.bin")?;
    println!("✓ 训练完成!");

    Ok(())
}
```

## 📊 格式对比

| 特性 | 二进制格式 (.bin) | JSON格式 (.json) |
|-----|------------------|------------------|
| 文件大小 | 小 | 大(约3-5倍) |
| 加载速度 | 快 | 较慢 |
| 人类可读 | ❌ | ✅ |
| 跨语言 | ❌ | ✅ |
| 优化器状态 | ✅ | ✅ |
| 适用场景 | 生产/训练 | 调试/研究 |

## 🗂️ 推荐的文件组织结构

```
your_project/
├── checkpoints/          # 训练checkpoint
│   ├── model_epoch_10.bin
│   ├── model_epoch_20.bin
│   └── ...
├── exports/              # 导出的模型
│   ├── model_v1.bin     # 生产模型
│   └── model_v1.json    # 调试用JSON
└── final_models/         # 最终发布模型
    └── model_release.bin
```

## ⚠️ 注意事项

1. **文件大小**: 模型文件可能较大(几MB到几GB),确保有足够磁盘空间

2. **版本兼容性**: 当前格式版本为v1,未来版本可能不向后兼容

3. **安全性**: 模型文件包含完整网络参数,妥善保管避免泄露

4. **JSON精度**: JSON格式可能有微小的浮点精度损失

5. **路径处理**: 建议创建目录后再保存:
   ```rust
   std::fs::create_dir_all("checkpoints")?;
   save_model_binary(&llm, "checkpoints/model.bin")?;
   ```

## 🐛 故障排查

### 问题: 加载模型失败

**可能原因:**
- 文件损坏
- 格式版本不匹配
- 文件路径错误

**解决方法:**
```rust
match load_model_binary("model.bin") {
    Ok(model) => println!("加载成功!"),
    Err(e) => eprintln!("加载失败: {}", e),
}
```

### 问题: 文件过大

**解决方法:**
- 使用二进制格式而非JSON
- 考虑模型压缩技术(剪枝、量化)
- 只保存必要的checkpoint

## 📖 相关API文档

### 保存函数

```rust
/// 保存模型到二进制文件
pub fn save_model_binary<P: AsRef<Path>>(
    model: &LLM,
    path: P,
) -> Result<(), Box<dyn std::error::Error>>

/// 保存模型到JSON文件
pub fn save_model_json<P: AsRef<Path>>(
    model: &LLM,
    path: P,
) -> Result<(), Box<dyn std::error::Error>>
```

### 加载函数

```rust
/// 从二进制文件加载模型
pub fn load_model_binary<P: AsRef<Path>>(
    path: P,
) -> Result<LLM, Box<dyn std::error::Error>>

/// 从JSON文件加载模型
pub fn load_model_json<P: AsRef<Path>>(
    path: P,
) -> Result<LLM, Box<dyn std::error::Error>>

/// 自动识别格式并加载
pub fn load_model_auto<P: AsRef<Path>>(
    path: P,
) -> Result<LLM, Box<dyn std::error::Error>>
```

## 🎯 最佳实践

1. **定期保存**: 训练过程中定期保存checkpoint,避免意外中断导致数据丢失

2. **命名规范**: 使用描述性的文件名,包含epoch数、日期等信息

3. **备份策略**: 保留多个历史版本,避免覆盖唯一的模型文件

4. **测试加载**: 保存后立即测试加载,确认文件完整性

5. **环境区分**: 开发环境使用JSON调试,生产环境使用二进制格式

## 🔬 高级用法

### 导出权重给Python使用

```rust
// 导出为JSON
save_model_json(&llm, "model_for_python.json")?;
```

```python
# Python代码读取
import json

with open("model_for_python.json", "r") as f:
    model_data = json.load(f)

# 访问权重
vocab = model_data["vocab"]
layers = model_data["layers"]
print(f"词汇量: {len(vocab['words'])}")
```

---

**相关文档:**
- [CLAUDE.md](CLAUDE.md) - 项目架构说明
- [README_zh.md](README_zh.md) - 项目介绍
