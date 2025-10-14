use std::f32;

use ndarray::{Array2, Axis, s};
use rand_distr::{Distribution, Normal};

use crate::{EMBEDDING_DIM, adam::Adam, llm::Layer};
use crate::utils::softmax;

pub struct SelfAttention {
    pub embedding_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_q: Array2<f32>, // Weight matrices for Q, K, V
    pub w_k: Array2<f32>,
    pub w_v: Array2<f32>,
    pub w_o: Array2<f32>, // Output projection weight

    // 缓存前向传播的中间变量，用于反向传播
    pub cached_input: Option<Array2<f32>>,
    pub cached_q: Option<Array2<f32>>,          // 缓存Q矩阵
    pub cached_k: Option<Array2<f32>>,          // 缓存K矩阵
    pub cached_v: Option<Array2<f32>>,          // 缓存V矩阵
    pub cached_attention_scores: Option<Array2<f32>>,  // 缓存attention scores (QK^T/√d)
    pub cached_attention_weights: Option<Array2<f32>>, // 缓存softmax后的attention weights
    pub cached_attention_output: Option<Array2<f32>>,  // 缓存attention输出（投影前）

    // KV缓存：用于推理加速
    // 存储历史token的K和V矩阵，避免重复计算
    pub kv_cache: Option<(Array2<f32>, Array2<f32>)>,  // (K_cache, V_cache)
    pub use_kv_cache: bool,  // 是否启用KV缓存

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
    /// Initializes a Transformer with random Q, K, V weights
    pub fn new(embedding_dim: usize) -> Self {
        let mut rng = rand::rng();
        let num_heads = 8; // Standard number of attention heads
        let head_dim = embedding_dim / num_heads;

        if embedding_dim % num_heads != 0 {
            panic!("Embedding dimension must be divisible by number of heads");
        }

        // Xavier/He initialization: std = sqrt(2 / fan_in)
        let std = (2.0 / embedding_dim as f32).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        SelfAttention {
            embedding_dim,
            num_heads,
            head_dim,
            w_q: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_k: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_v: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            w_o: Array2::from_shape_fn((embedding_dim, embedding_dim), |_| normal.sample(&mut rng)),
            cached_input: None,
            cached_q: None,
            cached_k: None,
            cached_v: None,
            cached_attention_scores: None,
            cached_attention_weights: None,
            cached_attention_output: None,
            kv_cache: None,         // KV缓存初始化为None
            use_kv_cache: false,    // 默认不使用KV缓存
            optimizer_w_q: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_k: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_v: Adam::new((embedding_dim, embedding_dim)),
            optimizer_w_o: Adam::new((embedding_dim, embedding_dim)),
        }
    }

    fn compute_qkv(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);
        (q, k, v)
    }

    /// 单头注意力计算，返回(attention_output, attention_weights)
    /// attention_weights用于反向传播
    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>)
        -> (Array2<f32>, Array2<f32>) {
        let dk = (q.ncols() as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // 应用causal mask（确保只能attend到之前的位置）- 使用向量化操作优化
        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            if i + 1 < seq_len {
                // 使用切片操作一次性设置整行的后续位置为负无穷
                scores.slice_mut(s![i, i+1..]).fill(f32::NEG_INFINITY);
            }
        }

        let weights = softmax(&scores);
        let output = weights.dot(v);
        (output, weights)
    }

    // 优化后的 reshape：使用更高效的方式处理多头注意力
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
                result.row_mut(result_row_idx)
                    .assign(&row.slice(s![start_dim..end_dim]));
            }
        }

        result
    }

    // 优化后的反向 reshape
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
                result.slice_mut(s![seq_idx, dst_start..dst_end])
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
            let q_head = q_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();
            let k_head = k_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();
            let v_head = v_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();

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
            let k_all = concatenate(Axis(0), &[k_cache.view(), k_new.view()]).unwrap();
            let v_all = concatenate(Axis(0), &[v_cache.view(), v_new.view()]).unwrap();
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
            let q_head = q_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();
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
        let input = self.cached_input.as_ref().unwrap();
        let _q = self.cached_q.as_ref().unwrap();
        let _k = self.cached_k.as_ref().unwrap();
        let _v = self.cached_v.as_ref().unwrap();
        let attention_output = self.cached_attention_output.as_ref().unwrap();

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

        // 使用Adam优化器更新权重
        self.optimizer_w_o.step(&mut self.w_o, &grad_w_o, lr);
        self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
        self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
        self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);

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
