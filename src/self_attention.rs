//! # 多头自注意力机制（Multi-Head Self-Attention）
//!
//! 这是 Transformer 架构的核心创新，让模型能够捕捉序列中的长距离依赖关系。
//!
//! ## 核心思想：注意力即"权重分配"
//!
//! 在自然语言中，不是所有词都同等重要。注意力机制让模型学习：
//! - 哪些词在理解当前词时更"相关"
//! - 如何动态调整对不同词的关注程度
//!
//! **直观理解**：
//! ```text
//! "我昨天在书店买了一本《机器学习》"
//!
//! 在理解"买"这个动作时，模型应该：
//! - 更多关注"我"（谁买？）→ 权重 0.5
//! - 适中关注"书店"（在哪里买？）→ 权重 0.3
//! - 少量关注"昨天"（什么时候？）→ 权重 0.2
//! ```
//!
//! ## 数学原理：缩放点积注意力（Scaled Dot-Product Attention）
//!
//! ### 公式
//! ```text
//! Attention(Q, K, V) = softmax(QK^T / √d_k) · V
//! ```
//!
//! **变量说明**：
//! - **Q（Query）**: "我在寻找什么" - 当前位置的查询向量
//! - **K（Key）**: "我能提供什么" - 所有位置的键向量（用于匹配）
//! - **V（Value）**: "我具体是什么" - 所有位置的值向量（用于加权）
//! - **d_k**: Key/Query 的维度（64），用于缩放防止梯度消失
//!
//! ### 步骤详解
//! 1. **计算相似度**: `Q·K^T` - 查询与每个键的点积（相似度越高，点积越大）
//! 2. **缩放**: `/ √d_k` - 防止点积过大导致 softmax 梯度消失
//! 3. **归一化**: `softmax` - 转换为概率分布（总和为1）
//! 4. **加权求和**: `·V` - 根据注意力权重加权各个值
//!
//! ## 多头机制
//!
//! **为什么需要多头？** 不同头学习不同类型的 attention：
//!
//! - **Head 1**: 语法关系（主谓宾结构）
//! - **Head 2**: 语义关系（同义词、反义词）
//! - **Head 3**: 位置关系（远近、顺序）
//! - **Head 4-8**: 其他抽象模式
//!
//! **实现方式**：
//! - 将 512 维分成 8 个头，每个头 64 维
//! - 并行计算 8 个注意力
//! - 最后拼接并投影回 512 维
//!
//! ## Causal Mask（因果掩码）
//!
//! **问题**：在生成文本时，模型不应该看到未来的词。
//!
//! **解决方案**：使用下三角矩阵将未来位置的注意力设为 -∞
//!
//! ```text
//! 注意力掩码矩阵：
//!    位置0 位置1 位置2 位置3
//! 0  [  √    ×    ×    ×  ]  ✓ 只能看自己
//! 1  [  √    √    ×    ×  ]  ✓ 可以看到位置0和1
//! 2  [  √    √    √    ×  ]  ✓ 可以看到位置0-2
//! 3  [  √    √    √    √  ]  ✓ 可以看到所有位置
//!
//! （× 表示设为 -∞，softmax 后概率为 0）
//! ```

use std::f32;

use ndarray::{Array2, Axis, s};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::utils::softmax;
use crate::{EMBEDDING_DIM, adam::Adam, llm::Layer};

/// **多头自注意力机制结构体**
pub struct SelfAttention {
    /// **嵌入维度**: 512（输入/输出的向量维度）
    pub embedding_dim: usize,

    /// **注意力头数**: 8（并行计算的注意力头数量）
    pub num_heads: usize,

    /// **每个头的维度**: 64（512 / 8 = 64）
    pub head_dim: usize,

    // ========== 核心权重矩阵 ==========
    /// **Query 投影矩阵** W_Q: (512, 512)
    /// 将输入转换为查询向量："我在寻找什么信息？"
    pub w_q: Array2<f32>,

    /// **Key 投影矩阵** W_K: (512, 512)
    /// 将输入转换为键向量："我能提供什么信息？"
    pub w_k: Array2<f32>,

    /// **Value 投影矩阵** W_V: (512, 512)
    /// 将输入转换为值向量："我具体包含什么内容？"
    pub w_v: Array2<f32>,

    /// **输出投影矩阵** W_O: (512, 512)
    /// 将多头拼接后的结果投影回原始维度
    pub w_o: Array2<f32>,

    // ========== 前向传播缓存（用于反向传播） ==========
    /// **缓存输入**: (seq_len, 512) - 原始输入，用于计算梯度
    pub cached_input: Option<Array2<f32>>,

    /// **缓存Q矩阵**: (seq_len, 512) - 查询矩阵
    pub cached_q: Option<Array2<f32>>,

    /// **缓存K矩阵**: (seq_len, 512) - 键矩阵
    pub cached_k: Option<Array2<f32>>,

    /// **缓存V矩阵**: (seq_len, 512) - 值矩阵
    pub cached_v: Option<Array2<f32>>,

    /// **缓存注意力分数**: (seq_len, seq_len) - QK^T/√d_k 的结果
    pub cached_attention_scores: Option<Array2<f32>>,

    /// **缓存注意力权重**: (seq_len, seq_len) - softmax 后的概率分布
    pub cached_attention_weights: Option<Array2<f32>>,

    /// **缓存注意力输出**: (seq_len, 512) - 多头拼接后、投影前的结果
    pub cached_attention_output: Option<Array2<f32>>,

    // ========== KV缓存优化（推理加速） ==========
    /// **KV缓存**: (K_cache, V_cache)
    ///
    /// 存储历史 token 的 K 和 V 矩阵，避免重复计算。
    ///
    /// **性能提升示例**：
    /// - 不使用缓存：生成100个token需要 O(100²) = 10,000 次计算
    /// - 使用缓存：生成100个token需要 O(100) = 100 次计算
    /// - **加速比**: 100倍！
    pub kv_cache: Option<(Array2<f32>, Array2<f32>)>,

    /// **是否启用KV缓存**
    /// - true: 推理模式（快速生成）
    /// - false: 训练模式（需要完整梯度）
    pub use_kv_cache: bool,

    /// **是否冻结注意力层参数更新**（用于稳定训练排障）
    pub freeze_updates: bool,

    // ========== Adam 优化器（每个权重矩阵一个） ==========
    pub optimizer_w_q: Adam,
    pub optimizer_w_k: Adam,
    pub optimizer_w_v: Adam,
    pub optimizer_w_o: Adam,
}

impl Default for SelfAttention {
    fn default() -> Self {
        SelfAttention::new(EMBEDDING_DIM)
    }
}

impl SelfAttention {
    /// **创建新的多头自注意力层**
    ///
    /// # 参数
    /// - `embedding_dim`: 嵌入维度（通常为512）
    ///
    /// # 架构配置
    /// - **头数**: 8个（Transformer 论文的标准配置）
    /// - **每头维度**: embedding_dim / num_heads = 64
    /// - **总参数量**: 4 × 512² = 1,048,576 参数（Q、K、V、O 四个矩阵）
    ///
    /// # 权重初始化
    /// 使用 He 初始化：std = sqrt(2 / embedding_dim)
    ///
    /// **为什么是 sqrt(2/512) ≈ 0.0625？**
    /// - 保持激活值的方差在层与层之间稳定
    /// - 防止梯度爆炸或消失
    ///
    /// # 示例
    /// ```rust
    /// let attention = SelfAttention::new(512);
    /// // 创建 8 头注意力，每头 64 维
    /// assert_eq!(attention.num_heads, 8);
    /// assert_eq!(attention.head_dim, 64);
    /// ```
    pub fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let num_heads = 8; // Transformer 标准：8个注意力头
        let head_dim = embedding_dim / num_heads;

        // 确保维度可以被头数整除
        let (num_heads, head_dim) = if embedding_dim % num_heads != 0 {
            log::warn!(
                "embedding_dim={} 不能被 num_heads={} 整除，回退为单头注意力",
                embedding_dim,
                num_heads
            );
            (1, embedding_dim)
        } else {
            (num_heads, head_dim)
        };

        // He 初始化：std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal_ok = Normal::new(0.0, std).ok();

        let w_q = if let Some(normal) = normal_ok.clone() {
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng))
        } else {
            log::warn!("SelfAttention: 正态分布初始化失败，W_q改用均匀分布");
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
                rng.random_range(-std..std)
            })
        };
        let w_k = if let Some(normal) = normal_ok.clone() {
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng))
        } else {
            log::warn!("SelfAttention: 正态分布初始化失败，W_k改用均匀分布");
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
                rng.random_range(-std..std)
            })
        };
        let w_v = if let Some(normal) = normal_ok.clone() {
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng))
        } else {
            log::warn!("SelfAttention: 正态分布初始化失败，W_v改用均匀分布");
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
                rng.random_range(-std..std)
            })
        };
        let w_o = if let Some(normal) = normal_ok {
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng))
        } else {
            log::warn!("SelfAttention: 正态分布初始化失败，W_o改用均匀分布");
            Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
                rng.random_range(-std..std)
            })
        };

        SelfAttention {
            embedding_dim,
            num_heads,
            head_dim,
            w_q,
            w_k,
            w_v,
            w_o,
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attention_scores: None,
            cached_attention_weights: None,
            cached_attention_output: None,
            kv_cache: None,      // 默认不使用 KV 缓存
            use_kv_cache: false, // 默认训练模式
            freeze_updates: false,
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_o: Adam::new((embedding_dim, embedding_dim)),
        }
    }

    /// **计算 Q、K、V 矩阵**
    ///
    /// 这是注意力机制的第一步：将输入投影到三个不同的"表示空间"。
    ///
    /// # 计算公式
    /// ```text
    /// Q = X · W_Q  (查询："我要找什么？")
    /// K = X · W_K  (键："我是什么？")
    /// V = X · W_V  (值："我的内容是什么？")
    /// ```
    ///
    /// # 参数
    /// - `input`: 输入张量 (seq_len, 512)
    ///
    /// # 返回值
    /// (Q, K, V) 三个矩阵，形状都是 (seq_len, 512)
    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);
        (q, k, v)
    }

    /// **单头注意力计算**
    ///
    /// 这是注意力机制的核心：通过 Q 和 K 的相似度，对 V 进行加权求和。
    ///
    /// # 算法步骤
    ///
    /// 1. **计算注意力分数**: scores = Q · K^T / √d_k
    ///    - 点积表示相似度
    ///    - 除以√d_k 防止值过大
    ///
    /// 2. **应用因果掩码**: 将未来位置设为 -∞
    ///    - 确保自回归性质（不能看到未来）
    ///
    /// 3. **Softmax 归一化**: weights = softmax(scores)
    ///    - 转换为概率分布（总和为1）
    ///
    /// 4. **加权求和**: output = weights · V
    ///    - 根据注意力权重组合值向量
    ///
    /// # 参数
    /// - `q`: Query 矩阵 (seq_len, head_dim=64)
    /// - `k`: Key 矩阵 (seq_len, head_dim=64)
    /// - `v`: Value 矩阵 (seq_len, head_dim=64)
    ///
    /// # 返回值
    /// - `output`: 注意力输出 (seq_len, head_dim)
    /// - `weights`: 注意力权重 (seq_len, seq_len)，用于反向传播
    fn attention(
        &self,
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // 步骤 1: 计算缩放点积注意力分数
        let dk = (q.ncols() as f32).sqrt(); // √d_k = √64 = 8
        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk; // (seq_len, seq_len)

        // 步骤 2: 应用因果掩码（Causal Mask）
        // 将未来位置设为 -∞，确保模型只能看到过去和当前的信息
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            if i + 1 < seq_len {
                // 使用切片操作一次性设置整行的后续位置为负无穷
                // 例如：位置0只能看自己，位置1可以看0和1，位置2可以看0、1、2
                scores.slice_mut(s![i, i + 1..]).fill(f32::NEG_INFINITY);
            }
        }

        // 步骤 3: Softmax 归一化
        // softmax 将 -∞ 转换为 0，其他值转换为 0-1 之间的概率
        let weights = softmax(&scores);

        // 步骤 4: 使用注意力权重加权 V
        let output = weights.dot(v);

        (output, weights)
    }

    /// **将矩阵重塑为多头格式**
    ///
    /// 这个函数将 (seq_len, 512) 的矩阵转换为多头格式 (seq_len×8, 64)。
    ///
    /// # 转换逻辑
    ///
    /// **输入**: (seq_len, 512) - 一个大的嵌入矩阵
    /// ```text
    /// [向量0: [d0, d1, ..., d511]]
    /// [向量1: [d0, d1, ..., d511]]
    /// ```
    ///
    /// **输出**: (seq_len×8, 64) - 8个头，每个头64维
    /// ```text
    /// Head 0: [向量0的d0-d63, 向量1的d0-d63, ...]
    /// Head 1: [向量0的d64-d127, 向量1的d64-d127, ...]
    /// ...
    /// Head 7: [向量0的d448-d511, 向量1的d448-d511, ...]
    /// ```
    ///
    /// # 示例
    /// ```text
    /// 输入: seq_len=2, embedding_dim=512
    /// [[向量0: 512维], [向量1: 512维]]
    ///
    /// 输出: seq_len×num_heads=16 行，每行 64 维
    /// 行0: 向量0的第0-63维 (Head 0)
    /// 行1: 向量0的第64-127维 (Head 1)
    /// ...
    /// 行7: 向量0的第448-511维 (Head 7)
    /// 行8: 向量1的第0-63维 (Head 0)
    /// ...
    /// ```
    fn reshape_for_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _embedding_dim) = x.dim();

        // 预分配结果矩阵
        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));

        // 优化的重排逻辑：减少内存访问次数
        for seq_idx in 0..seq_len {
            let row = x.row(seq_idx);
            for head_idx in 0..self.num_heads {
                let start_dim = head_idx * self.head_dim;
                let end_dim = start_dim + self.head_dim;
                let result_row_idx = seq_idx * self.num_heads + head_idx;

                // 使用切片赋值，比逐元素复制更高效
                result
                    .row_mut(result_row_idx)
                    .assign(&row.slice(s![start_dim..end_dim]));
            }
        }

        result
    }

    /// **将多头格式转换回正常矩阵**
    ///
    /// 这是 `reshape_for_heads` 的逆操作，将多头输出拼接回单个大矩阵。
    ///
    /// # 转换逻辑
    ///
    /// **输入**: (seq_len×8, 64) - 8个头的输出
    /// ```text
    /// 行0: 向量0_Head0 (64维)
    /// 行1: 向量0_Head1 (64维)
    /// ...
    /// 行7: 向量0_Head7 (64维)
    /// 行8: 向量1_Head0 (64维)
    /// ...
    /// ```
    ///
    /// **输出**: (seq_len, 512) - 拼接所有头
    /// ```text
    /// 向量0: [Head0的64维 | Head1的64维 | ... | Head7的64维] = 512维
    /// 向量1: [Head0的64维 | Head1的64维 | ... | Head7的64维] = 512维
    /// ```
    fn reverse_reshape_from_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len_times_heads, _head_dim) = x.dim();
        let seq_len = seq_len_times_heads / self.num_heads;

        let mut result = Array2::zeros((seq_len, self.num_heads * self.head_dim));

        // 优化的反向重排逻辑
        for seq_idx in 0..seq_len {
            for head_idx in 0..self.num_heads {
                let src_row_idx = seq_idx * self.num_heads + head_idx;
                let dst_start = head_idx * self.head_dim;
                let dst_end = dst_start + self.head_dim;

                // 使用切片赋值
                result
                    .slice_mut(s![seq_idx, dst_start..dst_end])
                    .assign(&x.row(src_row_idx));
            }
        }

        result
    }

    /// 多头自注意力的前向传播
    ///
    /// # 算法流程
    /// 1. 计算Q、K、V矩阵：Q=XW_q, K=XW_k, V=XW_v
    /// 2. 分割为多个注意力头 (num_heads=8)
    /// 3. 对每个头计算：Attention(Q,K,V) = softmax(QK^T/√d_k)V
    /// 4. 拼接所有头的输出
    /// 5. 通过输出投影：output = concat(heads)W_o
    ///
    /// # 参数
    /// - `input`: 输入张量，形状为 (seq_len, embedding_dim)
    ///
    /// # 返回
    /// 注意力输出，形状与输入相同
    fn multi_head_attention(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _embedding_dim) = input.dim();

        // 1. 计算Q, K, V并缓存
        let (q, k, v) = self.compute_qkv(input);
        self.cached_q = Some(q.clone());
        self.cached_k = Some(k.clone());
        self.cached_v = Some(v.clone());

        // 2. 分割为多个头
        let q_heads = self.reshape_for_heads(&q);
        let k_heads = self.reshape_for_heads(&k);
        let v_heads = self.reshape_for_heads(&v);

        // 3. 对每个头计算注意力
        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));

        for head in 0..self.num_heads {
            let q_head = q_heads
                .slice(s![head..seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();
            let k_head = k_heads
                .slice(s![head..seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();
            let v_head = v_heads
                .slice(s![head..seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();

            let (head_output, _head_weights) = self.attention(&q_head, &k_head, &v_head);

            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    result[[i * self.num_heads + head, j]] = head_output[[i, j]];
                }
            }
        }

        // 4. 合并所有头
        let combined = self.reverse_reshape_from_heads(&result);
        self.cached_attention_output = Some(combined.clone());

        // 5. 输出投影
        combined.dot(&self.w_o)
    }

    /// 启用KV缓存模式
    pub fn enable_kv_cache(&mut self) {
        self.use_kv_cache = true;
    }

    /// 禁用KV缓存模式并清空缓存
    pub fn disable_kv_cache(&mut self) {
        self.use_kv_cache = false;
        self.kv_cache = None;
    }

    /// 清空KV缓存（保持启用状态）
    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }

    /// 带KV缓存的多头自注意力前向传播
    ///
    /// # 算法优化
    /// 在自回归生成时，每次只生成一个新token。历史token的K和V矩阵不变，
    /// 可以直接从缓存中复用，只需计算新token的K和V。
    ///
    /// # 性能提升
    /// - 训练时：不使用缓存（需要完整的梯度）
    /// - 推理时：使用缓存，速度提升10-100倍
    ///
    /// # 参数
    /// - `input`: 输入张量，形状为 (seq_len, embedding_dim)
    ///   - 使用缓存时：seq_len=1（只有新token）
    ///   - 不使用缓存时：seq_len=任意值
    ///
    /// # 返回
    /// 注意力输出，形状与输入相同
    pub fn forward_with_kv_cache(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if !self.use_kv_cache {
            // 如果未启用KV缓存，使用标准的multi_head_attention
            return self.multi_head_attention(input);
        }

        let (seq_len, _embedding_dim) = input.dim();

        // 1. 计算新token的Q, K, V
        let (q_new, k_new, v_new) = self.compute_qkv(input);

        // 2. 合并KV缓存
        let (k_all, v_all) = if let Some((k_cache, v_cache)) = &self.kv_cache {
            // 如果有缓存，拼接新的K和V
            use ndarray::concatenate;
            let k_all = match concatenate(Axis(0), &[k_cache.view(), k_new.view()]) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("KV缓存拼接失败(K): {}，使用未缓存的K", e);
                    k_new.clone()
                }
            };
            let v_all = match concatenate(Axis(0), &[v_cache.view(), v_new.view()]) {
                Ok(v) => v,
                Err(e) => {
                    log::warn!("KV缓存拼接失败(V): {}，使用未缓存的V", e);
                    v_new.clone()
                }
            };
            (k_all, v_all)
        } else {
            // 如果没有缓存，直接使用新的K和V
            (k_new.clone(), v_new.clone())
        };

        // 3. 更新KV缓存
        self.kv_cache = Some((k_all.clone(), v_all.clone()));

        // 4. 分割为多个头
        let q_heads = self.reshape_for_heads(&q_new);
        let k_heads = self.reshape_for_heads(&k_all);
        let v_heads = self.reshape_for_heads(&v_all);

        // 5. 对每个头计算注意力
        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));

        for head in 0..self.num_heads {
            let q_head = q_heads
                .slice(s![head..seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();
            let k_head = k_heads.slice(s![head..; self.num_heads, ..]).to_owned();
            let v_head = v_heads.slice(s![head..; self.num_heads, ..]).to_owned();

            let (head_output, _head_weights) = self.attention(&q_head, &k_head, &v_head);

            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    result[[i * self.num_heads + head, j]] = head_output[[i, j]];
                }
            }
        }

        // 6. 合并所有头
        let combined = self.reverse_reshape_from_heads(&result);

        // 7. 输出投影
        combined.dot(&self.w_o)
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        self.multi_head_attention(input)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 获取缓存的前向传播中间变量
        let (Some(input), Some(_q), Some(_k), Some(_v), Some(attention_output)) = (
            self.cached_input.as_ref(),
            self.cached_q.as_ref(),
            self.cached_k.as_ref(),
            self.cached_v.as_ref(),
            self.cached_attention_output.as_ref(),
        ) else {
            log::warn!("SelfAttention.backward 在未执行 forward 的情况下被调用，直接传递梯度");
            return grads.clone();
        };

        // ========== 步骤1: 计算输出投影层的梯度 ==========
        // output = attention_output @ W_o
        // 因此: grad_W_o = attention_output^T @ grads
        let grad_w_o = attention_output.t().dot(grads);

        // grad_attention_output = grads @ W_o^T
        let grad_attention_output = grads.dot(&self.w_o.t());

        // ========== 步骤2: 通过注意力机制反向传播 ==========
        // 简化实现：直接将梯度传播回Q、K、V
        // 完整实现需要通过softmax和矩阵乘法反向传播，但这里使用简化版本

        // 对于 V: attention_output ≈ weights @ V
        // grad_V = weights^T @ grad_attention_output
        // 这里我们使用简化的近似，因为精确计算需要每个头的weights
        let grad_v = &grad_attention_output;

        // 对于 Q 和 K，梯度通过 scores = Q @ K^T / sqrt(d_k) 传播
        // 简化处理：假设梯度均匀分配
        let grad_q = &grad_attention_output;
        let grad_k = &grad_attention_output;

        // ========== 步骤3: 计算W_q, W_k, W_v的梯度并更新 ==========
        // Q = input @ W_q, 因此 grad_W_q = input^T @ grad_Q
        let grad_w_q = input.t().dot(grad_q);
        let grad_w_k = input.t().dot(grad_k);
        let grad_w_v = input.t().dot(grad_v);

        // 使用Adam优化器更新权重（可选冻结）
        if !self.freeze_updates {
            self.optimizer_w_o.step(&mut self.w_o, &grad_w_o, lr);
            self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
            self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
            self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);
        }

        // ========== 步骤4: 计算传播回输入的梯度 ==========
        // input的梯度来自Q、K、V三条路径
        // grad_input = grad_Q @ W_q^T + grad_K @ W_k^T + grad_V @ W_v^T
        let grad_input_from_q = grad_q.dot(&self.w_q.t());
        let grad_input_from_k = grad_k.dot(&self.w_k.t());
        let grad_input_from_v = grad_v.dot(&self.w_v.t());

        let grad_input = grad_input_from_q + grad_input_from_k + grad_input_from_v;

        grad_input
    }

    fn parameters(&self) -> usize {
        self.w_k.len() + self.w_q.len() + self.w_v.len() + self.w_o.len()
    }

    fn set_training_mode(&mut self, _training: bool) {}
}
