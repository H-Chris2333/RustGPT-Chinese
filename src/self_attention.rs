use std::f32;

use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};

use crate::{EMBEDDING_DIM, adam::Adam, llm::Layer};

pub struct SelfAttention {
    #[allow(dead_code)]
    embedding_dim: usize,
    num_heads: usize,
    head_dim: usize,
    w_q: Array2<f32>, // Weight matrices for Q, K, V
    w_k: Array2<f32>,
    w_v: Array2<f32>,
    w_o: Array2<f32>, // Output projection weight

    cached_input: Option<Array2<f32>>,
    cached_attention_weights: Option<Array2<f32>>, // Cache for backward pass

    optimizer_w_q: Adam,
    optimizer_w_k: Adam,
    optimizer_w_v: Adam,
    optimizer_w_o: Adam,
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
            cached_attention_weights: None,
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

    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let dk = (q.ncols() as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        let seq_len = scores.shape()[0];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[[i, j]] = f32::NEG_INFINITY;
            }
        }

        let weights = self.softmax(&scores);
        weights.dot(v)
    }

    fn softmax(&self, scores: &Array2<f32>) -> Array2<f32> {
        let mut result = scores.clone();

        // Apply softmax row-wise
        for mut row in result.rows_mut() {
            let max_val = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // Calculate exp for each element
            let exp_values: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exp_values.iter().sum();

            // Normalize by sum
            for (i, &exp_val) in exp_values.iter().enumerate() {
                row[i] = exp_val / sum_exp;
            }
        }

        result
    }

    // Function to reshape for multi-head attention
    fn reshape_for_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _embedding_dim) = x.dim();
        let _batch_size = seq_len;

        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));

        for i in 0..seq_len {
            for j in 0..self.num_heads {
                for k in 0..self.head_dim {
                    let orig_idx = j * self.head_dim + k;
                    result[[i * self.num_heads + j, k]] = x[[i, orig_idx]];
                }
            }
        }

        result
    }
    
    // Function to reverse reshape after multi-head attention
    fn reverse_reshape_from_heads(&self, x: &Array2<f32>) -> Array2<f32> {
        let (seq_len_times_heads, _head_dim) = x.dim();
        let seq_len = seq_len_times_heads / self.num_heads;

        let mut result = Array2::zeros((seq_len, self.num_heads * self.head_dim));

        for i in 0..seq_len {
            for j in 0..self.num_heads {
                for k in 0..self.head_dim {
                    let orig_idx = j * self.head_dim + k;
                    result[[i, orig_idx]] = x[[i * self.num_heads + j, k]];
                }
            }
        }

        result
    }

    fn softmax_backward(
        softmax_output: &Array2<f32>,
        grad_output: &Array2<f32>,
    ) -> Array2<f32> {
        let mut grad_input = softmax_output.clone();

        for ((mut grad_row, softmax_row), grad_out_row) in grad_input
            .outer_iter_mut()
            .zip(softmax_output.outer_iter())
            .zip(grad_output.outer_iter())
        {
            let dot = softmax_row
                .iter()
                .zip(grad_out_row.iter())
                .map(|(&y_i, &dy_i)| y_i * dy_i)
                .sum::<f32>();

            for ((g, &y_i), &dy_i) in grad_row
                .iter_mut()
                .zip(softmax_row.iter())
                .zip(grad_out_row.iter())
            {
                *g = y_i * (dy_i - dot);
            }
        }

        grad_input
    }
    
    // Perform multi-head attention
    fn multi_head_attention(&mut self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, _embedding_dim) = input.dim();

        let (q, k, v) = self.compute_qkv(input);

        let q_heads = self.reshape_for_heads(&q);
        let k_heads = self.reshape_for_heads(&k);
        let v_heads = self.reshape_for_heads(&v);

        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));

        for head in 0..self.num_heads {
            let q_head = q_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();
            let k_head = k_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();
            let v_head = v_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned();

            let head_output = self.attention(&q_head, &k_head, &v_head);

            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    result[[i * self.num_heads + head, j]] = head_output[[i, j]];
                }
            }
        }

        let combined = self.reverse_reshape_from_heads(&result);
        self.cached_attention_weights = Some(combined.clone());

        combined.dot(&self.w_o)
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        let attention_output = self.multi_head_attention(input);
        attention_output + input
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();

        let attention_weights = self.cached_attention_weights.as_ref().cloned().unwrap_or_else(|| input.clone());

        let grad_w_o = attention_weights.t().dot(grads);
        self.optimizer_w_o.step(&mut self.w_o, &grad_w_o, lr);

        let grad_after_proj = grads.dot(&self.w_o.t());

        let softmax_grad = Self::softmax_backward(&attention_weights, &grad_after_proj);

        // Compute gradients for Q, K, V weights using the cached input and gradients
        let grad_q = softmax_grad.dot(&self.w_q.t());
        let grad_k = softmax_grad.dot(&self.w_k.t());
        let grad_v = softmax_grad.dot(&self.w_v.t());

        let grad_w_q = input.t().dot(&grad_q);
        let grad_w_k = input.t().dot(&grad_k);
        let grad_w_v = input.t().dot(&grad_v);

        self.optimizer_w_q.step(&mut self.w_q, &grad_w_q, lr);
        self.optimizer_w_k.step(&mut self.w_k, &grad_w_k, lr);
        self.optimizer_w_v.step(&mut self.w_v, &grad_w_v, lr);

        let grad_input = grad_q + grad_k + grad_v;

        grad_input + grads
    }

    fn parameters(&self) -> usize {
        self.w_k.len() + self.w_q.len() + self.w_v.len() + self.w_o.len()
    }

    fn set_training_mode(&mut self, _training: bool) {}
}
