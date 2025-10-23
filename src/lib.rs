//! # RustGPT-Chinese - 从零实现的中文语言模型
//!
//! 这是一个纯 Rust 实现的 Transformer 语言模型教育项目。
//! 本项目的目标是帮助学习者深入理解 Transformer 架构和语言模型的工作原理。
//!
//! ## 核心设计理念
//!
//! 1. **纯 Rust 实现**：不依赖 PyTorch/TensorFlow，只使用 `ndarray` 进行矩阵运算
//! 2. **从零构建**：手动实现所有神经网络组件（注意力机制、前馈网络等）
//! 3. **教育优先**：代码注重可读性和教学性，而非生产性能
//! 4. **中文特化**：针对中文语言特性进行优化（jieba 分词、成语识别等）
//!
//! ## 模块组织
//!
//! ### 核心模型组件
//! - `llm`: 语言模型主类，包含训练和推理逻辑
//! - `transformer`: Transformer 块实现（注意力 + 前馈网络）
//! - `self_attention`: 多头自注意力机制的核心实现
//!
//! ### 神经网络层
//! - `embeddings`: 词嵌入层（token embedding + 位置编码）
//! - `feed_forward`: 前馈神经网络（FFN）
//! - `layer_norm`: 层归一化
//! - `dropout`: Dropout 正则化层
//! - `output_projection`: 输出投影层（映射到词汇表）
//!
//! ### 工具模块
//! - `vocab`: 词汇表管理和中文分词（jieba-rs）
//! - `adam`: Adam 优化器实现
//! - `dataset_loader`: 训练数据加载器
//! - `position_encoding`: 正弦位置编码
//! - `model_serialization`: 模型序列化和反序列化
//! - `performance_monitor`: 性能监控工具
//! - `utils`: 通用工具函数

// ============================================================================
// 模块声明
// ============================================================================

pub mod adam; // Adam 优化器：带动量的自适应学习率优化算法
pub mod dataset_loader; // 数据加载器：处理预训练和对话数据
pub mod dropout; // Dropout层：随机丢弃神经元，防止过拟合
pub mod embeddings; // 嵌入层：将token ID转换为稠密向量表示
pub mod feed_forward; // 前馈网络：Transformer中的全连接层部分
pub mod layer_norm; // 层归一化：稳定训练的归一化技术
pub mod llm; // 语言模型主类：整合所有组件的核心模型
pub mod model_serialization; // 模型序列化：保存和加载模型权重
pub mod output_projection; // 输出投影层：将隐藏状态映射到词汇表概率
pub mod performance_monitor;
pub mod position_encoding; // 位置编码：为序列注入位置信息
pub mod self_attention; // 自注意力机制：Transformer的核心组件
pub mod transformer; // Transformer块：注意力+前馈的完整模块
pub mod utils; // 工具函数：通用辅助函数
pub mod vocab; // 词汇表：管理token和ID的映射关系 // 性能监控：记录和分析训练/推理性能

// ============================================================================
// 重导出核心类型（简化外部使用）
// ============================================================================

pub use dataset_loader::{Dataset, DatasetType};
pub use embeddings::Embeddings;
pub use llm::{LLM, Layer};
pub use model_serialization::{
    load_model_auto, load_model_binary, load_model_json, save_model_binary, save_model_json,
};
pub use output_projection::OutputProjection;
pub use performance_monitor::PerformanceMonitor;
pub use transformer::TransformerBlock;
pub use utils::sample_normal;
pub use vocab::Vocab;

// ============================================================================
// 模型超参数（Model Hyperparameters）
// ============================================================================

/// **最大训练序列长度**
///
/// 定义模型在训练阶段能处理的最长 token 序列。对于中文，一个字通常是一个 token。
///
/// **为什么设置为 128？**
/// - 结合数据集规模与显存占用做出的折中
/// - 128 个 token 足以覆盖大多数训练样本
/// - 更长的序列会显著增加 O(n²) 的注意力计算成本
///
/// **实际应用**：
/// - 超过此长度的训练样本会被截断
/// - 在对话训练中，会保留最近的 128 个 token 作为上下文
pub const MAX_SEQ_LEN: usize = 128;

/// **最大推理序列长度（KV-Cache 窗口）**
///
/// 用于推理阶段 KV-Cache 的滑动窗口上限。大于训练长度，以便在推理时携带更长的上下文，
/// 同时又不会让缓存无限增长。
pub const MAX_INFERENCE_SEQ_LEN: usize = 2048;

/// **位置编码支持的最大长度**
///
/// 位置编码会在初始化时预计算 0..MAX_POSITIONAL_LEN-1 的所有位置值。该值与推理最大长度保持一致，
/// 以支持更长上下文的绝对位置索引。
pub const MAX_POSITIONAL_LEN: usize = MAX_INFERENCE_SEQ_LEN;

/// **嵌入维度 (Embedding Dimension)**
///
/// 每个 token 被表示为一个 256 维的向量。这是模型的 "内部表示空间" 的维度。
///
/// **为什么选择 256？**
/// - 对于小型教学模型，256 维能够在表达能力与计算量之间取得更好平衡
/// - 结合中文语料的特点（词/字混合），256 维足以区分常用词汇
/// - 更大的维度会显著增加自注意力和前馈层的乘法次数
///
/// **影响**：
/// - 维度越大，模型表达能力越强，但计算成本也越高
/// - 所有层的输入输出都必须匹配这个维度
pub const EMBEDDING_DIM: usize = 256;

/// **隐藏层维度 (Hidden Dimension)**
///
/// 前馈网络（FFN）中间层的维度。在 Transformer 中，FFN 通常是嵌入维度的 2-4 倍。
///
/// **为什么是 512？**
/// - 与 256 维的嵌入相匹配，保持 2× 的扩展倍率
/// - 有助于在有限计算预算内保留足够的非线性表达能力
/// - 更大的隐藏层会显著增加矩阵乘法成本
///
/// **作用**：
/// - 在 FFN 中：256 → 512 → 256（先扩展再压缩）
/// - 这种 "瓶颈结构" 帮助模型提取抽象特征
pub const HIDDEN_DIM: usize = 512;

/// **目标词汇表大小**
///
/// 理论上的词汇表上限。实际词汇表大小由训练数据动态构建。
///
/// **为什么是30000？**
/// - 常用中文字约3500个，加上词组、标点、英文，通常在5000-15000之间
/// - 30000是一个保守的上限，确保能容纳所有可能的词元
/// - 实际使用中，词汇表会根据训练数据动态构建，通常小于这个值
///
/// **注意**：
/// - 词汇表越大，输出层（output_projection）的参数就越多
/// - 本项目使用 jieba 分词，词汇量相对可控
pub const VOCAB_SIZE: usize = 30000;

// ============================================================================
// 数值稳定性常量 (Numerical Stability Constants)
// ============================================================================

/// **通用数值稳定性常量**
///
/// 用于避免除零错误和数值下溢。在层归一化、梯度计算等场景中使用。
///
/// **典型应用**：
/// ```rust
/// let variance = calculate_variance(x);
/// let normalized = x / (variance + EPSILON).sqrt(); // 避免除以0
/// ```
pub const EPSILON: f32 = 1e-8;

/// **对数运算专用常量**
///
/// 在计算交叉熵损失时，需要对概率取对数。由于 log(0) = -∞，
/// 我们需要确保输入始终大于零。
///
/// **应用场景**：
/// ```rust
/// let loss = -prob.max(LOG_EPSILON).ln(); // 避免 log(0)
/// ```
///
/// **为什么是1e-10？**
/// - 比 EPSILON 更小，因为对数函数在接近0时变化剧烈
/// - 足够小以不影响正常概率值（通常在 0.001-0.999 范围内）
pub const LOG_EPSILON: f32 = 1e-10;

/// **Softmax 归一化专用常量**
///
/// Softmax 函数将任意实数向量转换为概率分布。
/// 这个常量用于确保 softmax 输出始终大于0。
///
/// **应用示例**：
/// ```rust
/// let probs = softmax(logits);
/// let stable_probs = probs.mapv(|p| p.max(SOFTMAX_EPSILON)); // 确保概率>0
/// ```
///
/// **为什么是1e-12？**
/// - 需要比 LOG_EPSILON 更小，因为 softmax 输出可能非常接近0
/// - 对于低概率token（如罕见词），这个保护很重要
pub const SOFTMAX_EPSILON: f32 = 1e-12;
