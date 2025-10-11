use std::f32;

use ndarray::{Array2, Axis, s};
use rand_distr::{Distribution, Normal};

use crate::{EMBEDDING_DIM, adam::Adam, llm::Layer};

pub struct SelfAttention {
    pub embedding_dim: usize,
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
        let q = input.dot(&self.w_q); // Q = X * W_Q
        let k = input.dot(&self.w_k); // K = X * W_K
        let v = input.dot(&self.w_v); // V = X * W_V
        (q, k, v)
    }

    fn attention(&self, q: &Array2<f32>, k: &Array2<f32>, v: &Array2<f32>) -> Array2<f32> {
        let dk = (self.embedding_dim as f32).sqrt();

        let k_t = k.t();
        let mut scores = q.dot(&k_t) / dk;

        // Apply causal masking - prevent attention to future tokens
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
        let (seq_len, embedding_dim) = x.dim();
        let batch_size = seq_len;
        
        // Reshape from (seq_len, embedding_dim) to (seq_len, num_heads, head_dim)
        // We'll use a flattened representation: (seq_len * num_heads, head_dim)
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
        let (seq_len_times_heads, head_dim) = x.dim();
        let seq_len = seq_len_times_heads / self.num_heads;
        
        // Reshape from (seq_len * num_heads, head_dim) back to (seq_len, embedding_dim)
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
        softmax_output: &Array2<f32>, // shape: [seq_len, vocab_size]
        grad_output: &Array2<f32>,    // shape: [seq_len, vocab_size]
    ) -> Array2<f32> {
        let mut grad_input = softmax_output.clone(); // to hold the result

        for ((mut grad_row, softmax_row), grad_out_row) in grad_input
            .outer_iter_mut()
            .zip(softmax_output.outer_iter())
            .zip(grad_output.outer_iter())
        {
            // dot product: y ⊙ dL/dy
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
    fn multi_head_attention(&self, input: &Array2<f32>) -> Array2<f32> {
        let (seq_len, embedding_dim) = input.dim();
        
        // Calculate Q, K, V
        let q = input.dot(&self.w_q); // (seq_len, embedding_dim)
        let k = input.dot(&self.w_k); // (seq_len, embedding_dim)
        let v = input.dot(&self.w_v); // (seq_len, embedding_dim)
        
        // Reshape for multi-head attention
        let q_heads = self.reshape_for_heads(&q); // (seq_len * num_heads, head_dim)
        let k_heads = self.reshape_for_heads(&k); // (seq_len * num_heads, head_dim)
        let v_heads = self.reshape_for_heads(&v); // (seq_len * num_heads, head_dim)
        
        // Process each head separately
        let mut result = Array2::zeros((seq_len * self.num_heads, self.head_dim));
        
        for head in 0..self.num_heads {
            // Get the data for this head
            let q_head = q_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned(); // (seq_len, head_dim)
            let k_head = k_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned(); // (seq_len, head_dim)
            let v_head = v_heads.slice(s![head..seq_len * self.num_heads; self.num_heads, ..]).to_owned(); // (seq_len, head_dim)
            
            // Calculate attention for this head
            let dk = (self.head_dim as f32).sqrt();
            let k_head_t = k_head.t();
            let mut scores = q_head.dot(&k_head_t) / dk; // (seq_len, seq_len)
            
            // Apply causal masking - prevent attention to future tokens
            let scores_seq_len = scores.shape()[0];
            for i in 0..scores_seq_len {
                for j in (i + 1)..scores_seq_len {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
            }
            
            let weights = self.softmax(&scores); // (seq_len, seq_len)
            let head_output = weights.dot(&v_head); // (seq_len, head_dim)
            
            // Store the result for this head
            for i in 0..seq_len {
                for j in 0..self.head_dim {
                    result[[i * self.num_heads + head, j]] = head_output[[i, j]];
                }
            }
        }
        
        // Reverse reshape to combine heads
        let combined = self.reverse_reshape_from_heads(&result); // (seq_len, embedding_dim)
        
        // Apply output projection
        combined.dot(&self.w_o) // (seq_len, embedding_dim)
    }
}

impl Layer for SelfAttention {
    fn layer_type(&self) -> &str {
        "SelfAttention"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        self.cached_input = Some(input.clone());
        let attention_output = self.multi_head_attention(input);
        attention_output + input // residual connection (no LayerNorm here)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // For simplicity, we'll implement a simplified backward pass
        // A full implementation would require caching more intermediate values
        let input = self.cached_input.as_ref().unwrap();
        
        // For now, we'll just update the output weights and return gradients
        // In a more complete implementation, we would compute gradients for Q, K, V, and O matrices
        let attention_grads = grads; // Simplified: assume the gradient flows back through attention
        
        // Update output projection weights (simplified)
        let output_grads = attention_grads.t().dot(&(input.dot(&self.w_o)));
        self.optimizer_w_o.step(&mut self.w_o, &output_grads, lr);
        
        // Return gradient to propagate further back
        attention_grads + grads // Include residual connection
    }

    fn parameters(&self) -> usize {
        self.w_k.len() + self.w_q.len() + self.w_v.len() + self.w_o.len()
    }
}
