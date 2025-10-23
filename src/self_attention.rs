//! # 多头自注意力机制（Multi-Head Self-Attention）
//!
//! 这是 Transformer 架构的核心创新，让模型能够捕捉序列中的长距离依赖关系。
//!
//! ## 性能优化（v0.3.2）
//!
//! ### 1. 因果掩码缓存
//! - **问题**: 每次前向传播都需要逐元素填充 NEG_INFINITY 创建掩码矩阵
//! - **解决**: 使用 HashMap 缓存不同序列长度的掩码，避免重复创建
//! - **收益**: 减少 O(seq_len²) 的掩码创建开销
//!
//! ### 2. 优化矩阵乘法
//! - **策略**: 使用 ndarray 的优化 dot() 方法（基于 BLAS）
//! - **掩码应用**: 使用矩阵加法替代逐元素设置
//! - **并行处理**: 多头计算使用 rayon 并行化
//!
//! ### 3. 稳定的 Softmax 实现
//! - **数值稳定性**: 使用 log-sum-exp 技巧（减去最大值）
//! - **避免溢出**: 处理极大/极小值时保持数值稳定
//! - **梯度计算**: 简化但稳定的反向传播（注：完整梯度计算较复杂，当前使用近似）
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

use std::collections::HashMap;
use std::f32;

use ndarray::{s, Array2, Axis};

#[cfg(feature = "tensor-accel")]
use rayon::prelude::*;

use crate::{EMBEDDING_DIM, MAX_INFERENCE_SEQ_LEN, adam::Adam, llm::Layer, utils::sample_normal};

#[cfg(feature = "kv-cache")]
#[derive(Debug)]
struct KvCachePool {
    k: Array2<f32>,
    v: Array2<f32>,
    len: usize,
    capacity: usize,
}

#[cfg(feature = "kv-cache")]
impl KvCachePool {
    fn new(capacity: usize, embedding_dim: usize) -> Self {
        Self {
            k: Array2::zeros((capacity, embedding_dim)),
            v: Array2::zeros((capacity, embedding_dim)),
            len: 0,
            capacity,
        }
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

/// **稳定的 Softmax 实现（使用 log-sum-exp 技巧）**
///
/// 通过减去最大值来避免数值溢出，确保在处理大数值时的稳定性。
///
/// # 参数
/// - `logits`: 输入矩阵 (seq_len, seq_len)
///
/// # 返回值
/// Softmax 输出，形状与输入相同，每行元素和为1
fn stable_softmax(logits: &Array2<f32>) -> Array2<f32> {
    let mut result = Array2::zeros(logits.dim());

    for (i, row) in logits.rows().into_iter().enumerate() {
        // 找到该行的最大值（数值稳定性）
        let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // 计算 exp(x - max)
        let mut exp_vals = row.mapv(|x| (x - max_val).exp());

        // 计算归一化因子
        let sum_exp: f32 = exp_vals.sum();

        // 归一化（添加小的epsilon避免除零）
        if sum_exp > 1e-15 {
            exp_vals.mapv_inplace(|x| x / sum_exp);
        }

        result.row_mut(i).assign(&exp_vals);
    }

    result
}

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
    /// **KV缓存池**
    ///
    /// 存储历史 token 的 K 和 V 矩阵，避免重复计算。使用预分配的连续内存，
    /// 支持滑动窗口更新以限制历史长度。
    #[cfg(feature = "kv-cache")]
    pub kv_cache: Option<KvCachePool>,

    /// **KV缓存上限（滑动窗口）**
    ///
    /// 控制最多保留多少历史 token 的键值，以防止缓存无限增长。
    #[cfg(feature = "kv-cache")]
    pub kv_cache_limit: usize,

    /// **是否启用KV缓存**
    /// - true: 推理模式（快速生成）
    /// - false: 训练模式（需要完整梯度）
    pub use_kv_cache: bool,

    /// **是否冻结注意力层参数更新**（用于稳定训练排障）
    pub freeze_updates: bool,

    // ========== 因果掩码缓存（性能优化） ==========
    /// **缓存不同序列长度的因果掩码**
    /// Key: 序列长度, Value: 下三角掩码矩阵
    pub causal_mask_cache: HashMap<usize, Array2<f32>>,

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

        let w_q = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        let w_k = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        let w_v = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

        let w_o = Array2::from_shape_fn((embedding_dim, embedding_dim), |_| {
            sample_normal(&mut rng, 0.0, std)
        });

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
            #[cfg(feature = "kv-cache")]
            kv_cache: None,
            #[cfg(feature = "kv-cache")]
            kv_cache_limit: MAX_INFERENCE_SEQ_LEN,
            use_kv_cache: false, // 默认训练模式
            freeze_updates: false,
            causal_mask_cache: HashMap::new(), // 初始化掩码缓存
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_o: Adam::new((embedding_dim, embedding_dim)),
        }
    }

    /// **获取或创建因果掩码**
    ///
    /// 预生成并缓存下三角因果掩码，避免每次forward时逐元素填充。
    ///
    /// # 参数
    /// - `seq_len`: 序列长度
    ///
    /// # 返回值
    /// 因果掩码矩阵 (seq_len, seq_len)，下三角为0，上三角为-∞
    fn get_or_create_causal_mask(&mut self, seq_len: usize) -> &Array2<f32> {
        self.causal_mask_cache.entry(seq_len).or_insert_with(|| {
            let mut mask = Array2::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    mask[[i, j]] = f32::NEG_INFINITY;
                }
            }
            mask
        })
    }

    #[cfg(feature = "kv-cache")]
    fn ensure_kv_cache_pool(&mut self, required_capacity: usize) {
        let required_capacity = required_capacity.max(1);
        match self.kv_cache.as_mut() {
            Some(pool) => {
                if pool.capacity < required_capacity {
                    pool.k = Array2::zeros((required_capacity, self.embedding_dim));
                    pool.v = Array2::zeros((required_capacity, self.embedding_dim));
                    pool.capacity = required_capacity;
                    pool.len = pool.len.min(required_capacity);
                }
            }
            None => {
                self.kv_cache = Some(KvCachePool::new(required_capacity, self.embedding_dim));
            }
        }
    }

    #[cfg(feature = "kv-cache")]
    fn prepare_kv_cache(&mut self, incoming_len: usize) -> usize {
        self.ensure_kv_cache_pool(self.kv_cache_limit.max(incoming_len));
        let pool = self.kv_cache.as_mut().expect("kv-cache pool must exist");

        if pool.capacity == 0 {
            pool.len = 0;
            return 0;
        }

        if incoming_len >= pool.capacity {
            pool.len = 0;
            return 0;
        }

        if pool.len + incoming_len > pool.capacity {
            let overflow = pool.len + incoming_len - pool.capacity;
            if overflow >= pool.len {
                pool.len = 0;
            } else {
                let remaining = pool.len - overflow;
                let k_slice = pool.k.slice(s![overflow..pool.len, ..]).to_owned();
                let v_slice = pool.v.slice(s![overflow..pool.len, ..]).to_owned();
                pool.k
                    .slice_mut(s![0..remaining, ..])
                    .assign(&k_slice);
                pool.v
                    .slice_mut(s![0..remaining, ..])
                    .assign(&v_slice);
                pool.len = remaining;
            }
        }

        pool.len
    }

    fn build_sliding_mask(new_len: usize, base_len: usize, total_len: usize) -> Array2<f32> {
        let mut mask = Array2::zeros((new_len, total_len));
        for row in 0..new_len {
            let allowed = (base_len + row + 1).min(total_len);
            for col in allowed..total_len {
                mask[[row, col]] = f32::NEG_INFINITY;
            }
        }
        mask
    }

    fn compute_head_outputs(
        &self,
        q_heads: &Array2<f32>,
        k_heads: &Array2<f32>,
        v_heads: &Array2<f32>,
        q_seq_len: usize,
        k_seq_len: usize,
        mask: &Array2<f32>,
    ) -> Vec<Array2<f32>> {
        let compute = |head: usize| -> Array2<f32> {
            let q_head = q_heads
                .slice(s![head..q_seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();
            let k_head = k_heads
                .slice(s![head..k_seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();
            let v_head = v_heads
                .slice(s![head..k_seq_len * self.num_heads; self.num_heads, ..])
                .to_owned();
            let (head_output, _) = Self::attention_with_mask(&q_head, &k_head, &v_head, mask);
            head_output
        };

        #[cfg(feature = "tensor-accel")]
        {
            (0..self.num_heads)
                .into_par_iter()
                .map(|head| compute(head))
                .collect()
        }

        #[cfg(not(feature = "tensor-accel"))]
        {
            (0..self.num_heads).map(|head| compute(head)).collect()
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

    /// **单头注意力计算（带缓存掩码）**
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
    ///    - 使用预缓存的掩码矩阵
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
    /// - `mask`: 因果掩码 (seq_len, seq_len)
    ///
    /// # 返回值
    /// - `output`: 注意力输出 (seq_len, head_dim)
    /// - `weights`: 注意力权重 (seq_len, seq_len)，用于反向传播
    fn attention_with_mask(
        q: &Array2<f32>,
        k: &Array2<f32>,
        v: &Array2<f32>,
        mask: &Array2<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        // 步骤 1: 计算缩放点积注意力分数
        let dk = (q.ncols() as f32).sqrt(); // √d_k = √64 = 8

        // 使用 ndarray 的优化矩阵乘法
        let k_t = k.t();
        let scores = q.dot(&k_t) / dk; // (seq_len, seq_len)

        // 步骤 2: 应用因果掩码（直接相加，利用缓存的掩码）
        let masked_scores = scores + mask;

        // 步骤 3: Softmax 归一化（使用稳定的log-sum-exp实现）
        let weights = stable_softmax(&masked_scores);

        // 步骤 4: 使用注意力权重加权 V（优化矩阵乘法）
        let output = weights.dot(v);

        (output, weights)
    }

    /// **旧版单头注意力计算（保持向后兼容）**
    ///
    /// 这是注意力机制的核心：通过 Q 和 K 的相似度，对 V 进行加权求和。
    ///
    /// # 参数
    /// - `q`: Query 矩阵 (seq_len, head_dim=64)
    /// - `k`: Key 矩阵 (seq_len, head_dim=64)
    /// - `v`: Value 矩阵 (seq_len, head_dim=64)
    ///
    /// # 返回值
    /// - `output`: 注意力输出 (seq_len, head_dim)
    /// - `weights`: 注意力权重 (seq_len, seq_len)，用于反向传播
    fn attention(q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
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
        let weights = stable_softmax(&scores);

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

        for seq_idx in 0..seq_len {
            let row = x.row(seq_idx);
            for head_idx in 0..self.num_heads {
                let start_dim = head_idx * self.head_dim;
                let end_dim = start_dim + self.head_dim;
                let result_row_idx = seq_idx * self.num_heads + head_idx;

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

        for seq_idx in 0..seq_len {
            for head_idx in 0..self.num_heads {
                let src_row_idx = seq_idx * self.num_heads + head_idx;
                let dst_start = head_idx * self.head_dim;
                let dst_end = dst_start + self.head_dim;

                result
                    .slice_mut(s![seq_idx, dst_start..dst_end])
                    .assign(&x.row(src_row_idx));
            }
        }

        result
    }

    /// 多头自注意力的前向传播（优化版：使用缓存掩码）
    ///
    /// # 算法流程
    /// 1. 计算Q、K、V矩阵：Q=XW_q, K=XW_k, V=XW_v
    /// 2. 获取或创建因果掩码（缓存）
    /// 3. 分割为多个注意力头 (num_heads=8)
    /// 4. 对每个头计算：Attention(Q,K,V) = softmax(QK^T/√d_k)V
    /// 5. 拼接所有头的输出
    /// 6. 通过输出投影：output = concat(heads)W_o
    ///
    /// # 参数
    /// - `input`: 输入张量，形状为 (seq_len, embedding_dim)
    ///
    /// # 返回
    /// 注意力输出，形状与输入相同
    fn multi_head_attention(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _embedding_dim) = input.dim();
        if seq_len == 0 {
            return Array2::zeros((0, self.embedding_dim));
        }

        // 1. 计算 Q, K, V 并缓存（避免 clone，只在需要时缓存）
        let (q, k, v) = self.compute_qkv(input);

        // 2. 获取或创建因果掩码
        let mask = self.get_or_create_causal_mask(seq_len).clone();

        // 缓存 Q, K, V 用于反向传播
        self.cached_q = Some(q.clone());
        self.cached_k = Some(k.clone());
        self.cached_v = Some(v.clone());

        // 3. 分割为多个头
        let q_heads = self.reshape_for_heads(&q);
        let k_heads = self.reshape_for_heads(&k);
        let v_heads = self.reshape_for_heads(&v);

        // 4. 对每个头计算注意力（使用缓存掩码）
        let head_outputs =
            self.compute_head_outputs(&q_heads, &k_heads, &v_heads, seq_len, seq_len, &mask);

        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));
        for (head_idx, head_output) in head_outputs.iter().enumerate().take(self.num_heads) {
            for seq_idx in 0..seq_len {
                let row_idx = seq_idx * self.num_heads + head_idx;
                result.row_mut(row_idx).assign(&head_output.row(seq_idx));
            }
        }

        // 5. 合并所有头（避免不必要的 clone）
        let combined = self.reverse_reshape_from_heads(&result);
        self.cached_attention_output = Some(combined.clone());

        // 6. 输出投影
        combined.dot(&self.w_o)
    }

    /// 启用KV缓存模式
    pub fn enable_kv_cache(&mut self) {
        self.use_kv_cache = true;
        #[cfg(feature = "kv-cache")]
        {
            self.ensure_kv_cache_pool(self.kv_cache_limit);
        }
    }

    /// 禁用KV缓存模式并清空缓存
    pub fn disable_kv_cache(&mut self) {
        self.use_kv_cache = false;
        #[cfg(feature = "kv-cache")]
        {
            self.kv_cache = None;
        }
    }

    /// 清空KV缓存（保持启用状态）
    pub fn clear_kv_cache(&mut self) {
        #[cfg(feature = "kv-cache")]
        if let Some(pool) = self.kv_cache.as_mut() {
            pool.clear();
        }
    }

    /// 调整 KV 缓存窗口上限
    pub fn set_kv_cache_limit(&mut self, limit: usize) {
        #[cfg(feature = "kv-cache")]
        {
            let limit = limit.max(1);
            self.kv_cache_limit = limit;
            if let Some(pool) = self.kv_cache.as_mut() {
                if pool.capacity != limit {
                    pool.k = Array2::zeros((limit, self.embedding_dim));
                    pool.v = Array2::zeros((limit, self.embedding_dim));
                    pool.capacity = limit;
                    pool.len = pool.len.min(limit);
                }
            }
        }
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
        #[cfg(not(feature = "kv-cache"))]
        {
            return self.multi_head_attention(input);
        }

        #[cfg(feature = "kv-cache")]
        {
            if !self.use_kv_cache {
                return self.multi_head_attention(input);
            }

            let (seq_len, _embedding_dim) = input.dim();
            if seq_len == 0 {
                return Array2::zeros((0, self.embedding_dim));
            }

            // 1. 计算新 token 的 Q, K, V
            let (q_new, k_new, v_new) = self.compute_qkv(input);

            // 2. 准备缓存：裁剪/滑动窗口，返回插入前的历史长度
            let base_len = self.prepare_kv_cache(seq_len);
            let pool = self
                .kv_cache
                .as_mut()
                .expect("kv-cache pool must exist after preparation");

            let start = base_len;
            pool.k
                .slice_mut(s![start..start + seq_len, ..])
                .assign(&k_new);
            pool.v
                .slice_mut(s![start..start + seq_len, ..])
                .assign(&v_new);
            pool.len = start + seq_len;
            let total_len = pool.len;

            // 3. 构建滑动窗口掩码
            let mask = Self::build_sliding_mask(seq_len, base_len, total_len);

            // 4. 将缓存展平为按头拆分的矩阵
            let k_all = pool.k.slice(s![0..total_len, ..]).to_owned();
            let v_all = pool.v.slice(s![0..total_len, ..]).to_owned();

            self.cached_q = Some(q_new.clone());
            self.cached_k = Some(k_all.clone());
            self.cached_v = Some(v_all.clone());

            let q_heads = self.reshape_for_heads(&q_new);
            let k_heads = self.reshape_for_heads(&k_all);
            let v_heads = self.reshape_for_heads(&v_all);

            let head_outputs = self.compute_head_outputs(
                &q_heads,
                &k_heads,
                &v_heads,
                seq_len,
                total_len,
                &mask,
            );

            let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));
            for (head_idx, head_output) in head_outputs.iter().enumerate().take(self.num_heads) {
                for seq_idx in 0..seq_len {
                    let row_idx = seq_idx * self.num_heads + head_idx;
                    result.row_mut(row_idx).assign(&head_output.row(seq_idx));
                }
            }

            let combined = self.reverse_reshape_from_heads(&result);
            self.cached_attention_output = Some(combined.clone());

            combined.dot(&self.w_o)
        }
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

    fn forward_inference(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        if self.use_kv_cache {
            self.forward_with_kv_cache(input)
        } else {
            self.multi_head_attention(input)
        }
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

        grad_input_from_q + grad_input_from_k + grad_input_from_v
    }

    fn parameters(&self) -> usize {
        self.w_k.len() + self.w_q.len() + self.w_v.len() + self.w_o.len()
    }

    fn reset_inference_cache(&mut self) {
        #[cfg(feature = "kv-cache")]
        {
            self.clear_kv_cache();
        }
    }

    fn set_inference_cache_limit(&mut self, max_len: usize) {
        self.set_kv_cache_limit(max_len);
    }

    fn set_training_mode(&mut self, training: bool) {
        if training {
            self.use_kv_cache = false;
            #[cfg(feature = "kv-cache")]
            {
                self.clear_kv_cache();
            }
        } else {
            self.use_kv_cache = true;
        }
    }
}
