# 🚀 如何保存和加载模型

## 📖 快速开始

现在程序已经内置了**交互式模型管理功能**!运行程序时会自动引导您完成模型的保存和加载。

### 使用方法

```bash
cargo run
```

## 🎯 功能演示

### 场景1: 首次运行(训练新模型)

```
╔═══════════════════════════════════════════════════════════╗
║         RustGPT-Chinese - 中文GPT模型训练系统            ║
╚═══════════════════════════════════════════════════════════╝

📝 未检测到已保存的模型，将开始训练新模型...

[开始训练...]

╔═══════════════════════════════════════════════════════════╗
║                    模型保存选项                           ║
╚═══════════════════════════════════════════════════════════╝

是否保存当前模型? (y/n): y

选择保存格式:
  1) 二进制格式 (.bin) - 推荐，文件小、速度快
  2) JSON格式 (.json) - 人类可读，便于调试
  3) 两种格式都保存

请选择 (1/2/3): 1

文件名 (默认: checkpoints/model_final.bin):
✅ 模型已保存: checkpoints/model_final.bin
```

### 场景2: 加载已有模型

```bash
cargo run
```

```
╔═══════════════════════════════════════════════════════════╗
║         RustGPT-Chinese - 中文GPT模型训练系统            ║
╚═══════════════════════════════════════════════════════════╝

🔍 检测到已保存的模型:
  ✓ checkpoints/model_final.bin
  ✓ checkpoints/model_pretrained.bin

是否加载已有模型? (y/n): y

选择要加载的模型:
  1) checkpoints/model_final.bin (最终模型)
  2) checkpoints/model_pretrained.bin (预训练checkpoint)
请选择 (1/2): 1

📂 正在加载模型: checkpoints/model_final.bin...
   [1] 重建层... ✓
   [2] 重建层... ✓
   [3] 重建层... ✓
   [4] 重建层... ✓
   [5] 重建层... ✓
   [6] 重建层... ✓

✅ 模型加载成功!
  • 词汇量: 1234
  • 总参数: 36123456
  • 网络架构: Embeddings, TransformerBlock, ...

是否继续训练此模型? (y/n): n

✓ 跳过训练，直接进入交互模式
```

### 场景3: 断点续训

```
是否继续训练此模型? (y/n): y

🔄 继续训练模式

训练轮数 (默认50): 30
学习率 (默认0.0001): 0.00005

开始继续训练 (30 epochs, lr=0.00005)...

[训练过程...]

✅ 继续训练完成!
```

### 场景4: 交互模式中保存

```
╔═══════════════════════════════════════════════════════════╗
║                    交互模式                               ║
╚═══════════════════════════════════════════════════════════╝

💡 输入问题后按回车生成回答
💡 输入 'exit' 退出程序
💡 输入 'clear' 清空对话上下文
💡 输入 'save' 保存当前模型

👤 用户: save

选择保存格式:
  1) 二进制格式 (.bin) - 推荐，文件小、速度快
  2) JSON格式 (.json) - 人类可读，便于调试
  3) 两种格式都保存

请选择 (1/2/3): 3

保存二进制格式...
✓ 二进制格式已保存: checkpoints/model_final.bin
保存JSON格式...
✓ JSON格式已保存: exports/model_final.json

👤 用户:
```

## 📂 文件组织结构

程序会自动创建以下目录结构:

```
RustGPT-Chinese/
├── checkpoints/              # 二进制格式checkpoint
│   ├── model_pretrained.bin # 预训练checkpoint
│   └── model_final.bin      # 最终训练模型
├── exports/                  # JSON格式导出
│   └── model_final.json     # 用于调试和分析
└── data/                     # 训练数据
    ├── pretraining_data.json
    └── chat_training_data.json
```

## 🎨 保存格式对比

### 二进制格式 (.bin)
- ✅ **文件小** - 约95MB(默认模型)
- ✅ **加载快** - 1-2秒
- ✅ **完整状态** - 包含优化器动量
- ✅ **推荐用于**: 日常训练、生产部署
- ❌ 不可人工查看

### JSON格式 (.json)
- ✅ **人类可读** - 可直接查看权重
- ✅ **跨语言** - Python可读取
- ✅ **调试友好** - 方便检查参数
- ✅ **推荐用于**: 调试、研究、跨平台共享
- ❌ 文件大(约300-400MB)
- ❌ 加载较慢

## 💡 使用技巧

### 1. 定期保存checkpoint

在预训练阶段结束时,程序会询问:

```
💾 是否保存预训练checkpoint? (y/n): y
✓ 预训练checkpoint已保存
```

**建议**: 总是保存预训练checkpoint,这样可以从预训练阶段继续微调。

### 2. 使用不同的文件名

保存时可以自定义文件名:

```
文件名 (默认: checkpoints/model_final.bin): checkpoints/model_epoch_200.bin
✅ 模型已保存: checkpoints/model_epoch_200.bin
```

### 3. 导出为JSON供Python使用

```bash
# 在交互模式中输入
save

# 选择格式
请选择 (1/2/3): 2

# 指定文件名
文件名 (默认: exports/model_final.json): exports/model_for_analysis.json
```

然后在Python中:

```python
import json

with open("exports/model_for_analysis.json", "r") as f:
    model = json.load(f)

print(f"词汇量: {len(model['vocab']['words'])}")
print(f"网络层数: {len(model['layers'])}")
# 分析权重...
```

### 4. 交互命令

在交互模式下可用的命令:

- `save` - 保存当前模型
- `clear` - 清空对话上下文
- `exit` - 退出程序
- 其他任何输入 - 与模型对话

## 🔧 高级用法

### 编程方式使用API

如果您想在代码中直接使用保存/加载功能:

```rust
use llm::{LLM, save_model_binary, load_model_binary, save_model_json};

// 保存模型
let llm = LLM::default();
save_model_binary(&llm, "my_model.bin")?;

// 加载模型
let loaded_llm = load_model_binary("my_model.bin")?;

// 保存为JSON
save_model_json(&llm, "my_model.json")?;
```

### 使用示例程序

项目还包含一个独立的示例程序:

```bash
cargo run --example model_persistence save      # 训练并保存
cargo run --example model_persistence load      # 加载并使用
cargo run --example model_persistence continue  # 继续训练
```

## ⚠️ 注意事项

1. **磁盘空间**: 确保有足够空间(二进制约100MB,JSON约400MB)

2. **训练时间**: 完整训练需要较长时间,建议:
   - 首次训练时保存预训练checkpoint
   - 可以从checkpoint继续,避免重复训练

3. **版本兼容**: 当前版本为v1,未来版本可能不向后兼容

4. **路径处理**: 程序会自动创建 `checkpoints/` 和 `exports/` 目录

## 🐛 常见问题

**Q: 加载模型失败怎么办?**

A: 检查:
- 文件路径是否正确
- 文件是否完整(没有损坏)
- 如果是跨版本加载,可能不兼容

**Q: 可以只保存模型权重而不保存优化器状态吗?**

A: 当前版本两种格式都会保存优化器状态。如果只需要权重,建议在代码中修改序列化逻辑。

**Q: 如何查看模型文件大小?**

A: 在交互模式下:
```bash
# 在另一个终端
ls -lh checkpoints/
ls -lh exports/
```

## 📚 相关文档

- [model_persistence_guide_zh.md](model_persistence_guide_zh.md) - 详细API文档
- [CLAUDE.md](../CLAUDE.md) - 项目架构说明
- [README_zh.md](../README_zh.md) - 项目介绍

---

**祝您使用愉快! 🎉**
