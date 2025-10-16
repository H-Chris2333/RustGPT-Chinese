use std::cmp::Ordering;

use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand::rng;

use crate::utils::softmax;
use crate::{
    EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LOG_EPSILON, MAX_SEQ_LEN, SOFTMAX_EPSILON, Vocab,
    output_projection::OutputProjection, transformer::TransformerBlock,
};
pub trait Layer {
    fn layer_type(&self) -> &str;

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32>;

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32>;

    fn parameters(&self) -> usize;

    fn set_training_mode(&mut self, _training: bool) {}
}

#[allow(clippy::upper_case_acronyms)]
pub struct LLM {
    pub vocab: Vocab,
    pub network: Vec<Box<dyn Layer>>,
    pub context_window: Vec<usize>,
    pub max_context_length: usize,
    pub training: bool,
}

/// 早停机制
///
/// 监控训练loss，如果长时间不改善则自动停止训练
pub struct EarlyStopping {
    /// 容忍多少个epoch loss不改善
    patience: usize,

    /// 当前最佳loss
    best_loss: f32,

    /// 已经多少个epoch没有改善
    counter: usize,

    /// 最小改善幅度（小于这个值不算改善）
    min_delta: f32,

    /// 最佳模型所在的epoch
    best_epoch: usize,
}

impl EarlyStopping {
    /// 创建早停监控器
    ///
    /// # 参数
    /// - `patience`: 容忍epoch数（推荐30-50）
    /// - `min_delta`: 最小改善幅度（推荐0.001）
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            best_loss: f32::INFINITY,
            counter: 0,
            min_delta,
            best_epoch: 0,
        }
    }

    /// 检查是否应该停止训练
    ///
    /// # 返回值
    /// - `true`: 应该停止训练
    /// - `false`: 继续训练
    pub fn should_stop(&mut self, current_loss: f32, current_epoch: usize) -> bool {
        // 如果loss有明显改善
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.best_epoch = current_epoch;
            self.counter = 0;
            false
        } else {
            // loss没有改善
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// 获取最佳loss和对应的epoch
    pub fn best_state(&self) -> (f32, usize) {
        (self.best_loss, self.best_epoch)
    }
}

impl Default for LLM {
    fn default() -> Self {
        let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let transformer_block_4 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
        let output_projection = OutputProjection::new(EMBEDDING_DIM, Vocab::default_words().len());
        Self {
            vocab: Vocab::default(),
            network: vec![
                Box::new(Embeddings::default()),
                Box::new(transformer_block_1),
                Box::new(transformer_block_2),
                Box::new(transformer_block_3),
                Box::new(transformer_block_4),
                Box::new(output_projection),
            ],
            context_window: Vec::new(),
            max_context_length: MAX_SEQ_LEN,
            training: true,
        }
    }
}

impl LLM {
    pub fn new(vocab: Vocab, network: Vec<Box<dyn Layer>>) -> Self {
        Self {
            vocab,
            network,
            context_window: Vec::new(),
            max_context_length: MAX_SEQ_LEN,
            training: true,
        }
    }
}

impl LLM {
    pub fn network_description(&self) -> String {
        self.network
            .iter()
            .map(|layer| layer.layer_type())
            .collect::<Vec<&str>>()
            .join(", ")
    }

    pub fn total_parameters(&self) -> usize {
        self.network
            .iter()
            .map(|layer| layer.parameters())
            .sum::<usize>()
    }

    pub fn set_training_mode(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.network {
            layer.set_training_mode(training);
        }
    }

    #[allow(dead_code)]
    pub fn predict(&mut self, text: &str) -> String {
        self.predict_with_sampling(text, 1.0, 0.9, 5)
    }

    #[allow(dead_code)]
    pub fn predict_with_sampling(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        self.forward_with_sampling(text, temperature, top_p, top_k)
    }

    pub fn predict_with_context(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        // Tokenize the new input
        let new_tokens = self.tokenize(text);

        // Combine context with new input
        let mut all_tokens = self.context_window.clone();
        all_tokens.extend_from_slice(&new_tokens);

        // Ensure we don't exceed the maximum sequence length
        if all_tokens.len() > MAX_SEQ_LEN {
            let start_idx = all_tokens.len() - MAX_SEQ_LEN;
            all_tokens = all_tokens[start_idx..].to_vec();
        }

        let result = self.generate_with_context(&all_tokens, temperature, top_p, top_k);

        self.add_to_context(&new_tokens);
        let result_tokens = self.tokenize(&result);
        self.add_to_context(&result_tokens);

        result
    }

    pub fn predict_with_beam_search(
        &mut self,
        text: &str,
        beam_width: usize,
        max_length: usize,
    ) -> String {
        self.beam_search(text, beam_width, max_length)
    }

    #[allow(dead_code)]
    fn forward_with_sampling(
        &mut self,
        text: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        if tokenized.is_empty() {
            return String::new();
        }

        let input_len = tokenized.len();

        if input_len >= MAX_SEQ_LEN {
            return String::new();
        }

        for _ in 0..(MAX_SEQ_LEN - input_len) {
            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            let probs = softmax(&last_logit);

            let adjusted_probs = Self::apply_temperature(&probs, temperature);

            let tokens = if top_k > 0 {
                self.top_k_sampling(&adjusted_probs, top_k)
            } else {
                self.top_p_sampling(&adjusted_probs, top_p)
            };

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        let token_strs = output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>();

        let raw_output = token_strs.join(" ");
        self.post_process_chinese_text(&raw_output)
    }

    #[allow(dead_code)]
    fn forward(&mut self, text: &str) -> Vec<usize> {
        let mut tokenized = self.tokenize(text);
        let mut output_tokens: Vec<usize> = Vec::new();

        if tokenized.is_empty() {
            return output_tokens;
        }

        let input_len = tokenized.len();

        if input_len >= MAX_SEQ_LEN {
            return output_tokens;
        }

        for _ in 0..(MAX_SEQ_LEN - input_len) {
            if output_tokens.len() >= MAX_SEQ_LEN - 1 {
                break;
            }

            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            let probs = softmax(&last_logit);

            let tokens = Self::greedy_decode(&probs);

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        output_tokens
    }

    pub fn train(&mut self, data: Vec<&str>, epochs: usize, initial_lr: f32) {
        self.set_training_mode(true);

        let tokenized_data = data
            .iter()
            .map(|input| self.tokenize(input))
            .collect::<Vec<Vec<usize>>>();

        for epoch in 0..epochs {
            let decay_rate: f32 = 0.95;
            let decay_steps = 10.0;
            let current_lr = initial_lr * decay_rate.powf(epoch as f32 / decay_steps);

            let mut total_loss = 0.0;
            for training_row in &tokenized_data {
                if training_row.len() < 2 {
                    continue;
                }

                // 1. Slice input and targets
                let input_ids = &training_row[..training_row.len() - 1]; // Exclude the last token
                let target_ids = &training_row[1..]; // This is a vector. Each element is the index in the vocab. 

                // Forward pass
                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                let probs = softmax(&logits);

                total_loss += Self::cross_entropy_loss_step(&probs, target_ids);

                // Backward pass
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids); // this is d_L/d_output_projection

                Self::clip_gradients(&mut grads_output, 5.0);

                for layer in self.network.iter_mut().rev() {
                    grads_output = layer.backward(&grads_output, current_lr);
                }

                let tokens = Self::greedy_decode(&probs);
                let next_token = tokens[tokens.len() - 1];

                if next_token == self.vocab.encode("</s>").unwrap() {
                    continue;
                }
            }

            println!(
                "Epoch {}: Loss = {:.4}, LR = {:.6}",
                epoch,
                total_loss / tokenized_data.len() as f32,
                current_lr
            );
        }

        self.set_training_mode(false);
    }

    // ═════════════════════════════════════════════════════════════════════════════
    // 🚀 阶段1训练优化 - 性能优化方法
    // ═════════════════════════════════════════════════════════════════════════════

    /// 余弦退火学习率调度（带重启）
    ///
    /// # 参数
    /// - `initial_lr`: 初始学习率（如 0.001）
    /// - `epoch`: 当前epoch
    /// - `total_epochs`: 总epoch数
    /// - `num_restarts`: 重启次数（如2表示训练分为3个周期）
    ///
    /// # 示例
    /// ```rust
    /// // 500 epochs, 2次重启，每个周期约166 epochs
    /// let lr = LLM::cosine_annealing_lr(0.001, epoch, 500, 2);
    /// ```
    pub fn cosine_annealing_lr(
        initial_lr: f32,
        epoch: usize,
        total_epochs: usize,
        num_restarts: usize,
    ) -> f32 {
        // 计算每个周期的长度
        let cycle_length = total_epochs / (num_restarts + 1);

        // 当前在周期内的位置
        let cycle_epoch = epoch % cycle_length;

        // 周期内的进度 [0, 1]
        let progress = cycle_epoch as f32 / cycle_length as f32;

        // 最小学习率为初始值的1%
        let min_lr = initial_lr * 0.01;

        // 余弦退火公式
        min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
    }

    /// 计算梯度L2范数
    fn compute_grad_norm(grads: &Array2<f32>) -> f32 {
        grads.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// 完整优化的训练方法（集成所有阶段1优化）
    ///
    /// # 优化特性
    /// - ✅ 数据预处理缓存（避免重复tokenization）
    /// - ✅ 余弦退火学习率调度
    /// - ✅ 早停机制
    /// - ✅ 增强训练监控（困惑度、梯度范数、训练速度）
    /// - ✅ 梯度累积（模拟大batch训练）
    ///
    /// # 参数
    /// - `data`: 训练数据
    /// - `max_epochs`: 最大epoch数
    /// - `initial_lr`: 初始学习率
    /// - `patience`: 早停容忍epoch数
    /// - `accumulation_steps`: 梯度累积步数（推荐4-8）
    ///
    /// # 返回值
    /// 实际训练的epoch数
    pub fn train_monitored(
        &mut self,
        data: Vec<&str>,
        max_epochs: usize,
        initial_lr: f32,
        patience: usize,
        accumulation_steps: usize,
    ) -> usize {
        self.set_training_mode(true);

        // 🔥 优化1：数据预处理缓存（一次性tokenize，避免重复计算）
        println!("📝 正在预处理训练数据...");
        let start_time = std::time::Instant::now();
        let tokenized_data: Vec<Vec<usize>> =
            data.iter().map(|input| self.tokenize(input)).collect();
        println!(
            "✅ 数据预处理完成，共 {} 个序列（耗时 {:.2}s）",
            tokenized_data.len(),
            start_time.elapsed().as_secs_f32()
        );

        let mut early_stopping = EarlyStopping::new(patience, 0.001);
        let training_start_time = std::time::Instant::now();

        for epoch in 0..max_epochs {
            let epoch_start = std::time::Instant::now();

            // 🔥 优化2：余弦退火学习率调度
            let current_lr = Self::cosine_annealing_lr(initial_lr, epoch, max_epochs, 2);

            let mut total_loss = 0.0;
            let mut total_grad_norm = 0.0;
            let mut sample_count = 0;

            // 🔥 优化5：梯度累积相关变量
            let mut accumulated_grads: Option<Array2<f32>> = None;
            let mut step_count = 0;

            for (idx, training_row) in tokenized_data.iter().enumerate() {
                if training_row.len() < 2 {
                    continue;
                }

                // 前向传播
                let input_ids = &training_row[..training_row.len() - 1];
                let target_ids = &training_row[1..];

                let mut input: Array2<f32> = Array2::zeros((1, input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x| x as f32).collect::<Array1<f32>>());

                for layer in &mut self.network {
                    input = layer.forward(&input);
                }

                let logits = input;
                let probs = softmax(&logits);
                total_loss += Self::cross_entropy_loss_step(&probs, target_ids);

                // 计算梯度但不立即更新
                let mut grads_output = Self::compute_gradients_step(&probs, target_ids);

                // 🔥 优化4：记录梯度范数（用于监控）
                total_grad_norm += Self::compute_grad_norm(&grads_output);

                Self::clip_gradients(&mut grads_output, 5.0);

                // 🔥 优化5：梯度累积逻辑（修复形状兼容性问题）
                // 由于变长序列导致梯度形状不同，只在形状兼容时进行累积
                if accumulated_grads.is_none() {
                    accumulated_grads = Some(grads_output.clone());
                    step_count += 1;
                } else if let Some(ref mut acc_grads) = accumulated_grads {
                    // 检查形状是否兼容
                    if acc_grads.shape() == grads_output.shape() {
                        *acc_grads = &*acc_grads + &grads_output;
                        step_count += 1;
                    } else {
                        // 形状不兼容时，直接使用当前梯度更新参数并重置累积
                        let mut current_grad = grads_output;
                        for layer in self.network.iter_mut().rev() {
                            current_grad = layer.backward(&current_grad, current_lr);
                        }
                        accumulated_grads = None;
                        step_count = 0;
                    }
                }

                // 每accumulation_steps步或最后一个样本时更新参数
                let should_update =
                    step_count >= accumulation_steps || idx == tokenized_data.len() - 1;

                if should_update && accumulated_grads.is_some() {
                    if let Some(mut acc_grads) = accumulated_grads.take() {
                        // 平均梯度（重要！）
                        acc_grads.mapv_inplace(|x| x / step_count as f32);

                        // 反向传播更新参数
                        let mut current_grad = acc_grads;
                        for layer in self.network.iter_mut().rev() {
                            current_grad = layer.backward(&current_grad, current_lr);
                        }
                    }

                    step_count = 0;
                }

                sample_count += 1;
            }

            let epoch_time = epoch_start.elapsed().as_secs_f32();
            let avg_loss = total_loss / sample_count as f32;
            let avg_grad_norm = total_grad_norm / sample_count as f32;
            let perplexity = avg_loss.exp();
            let samples_per_sec = sample_count as f32 / epoch_time;

            // 🔥 优化4：增强训练监控输出
            if epoch % 10 == 0 || epoch == max_epochs - 1 {
                let progress = (epoch + 1) as f32 / max_epochs as f32 * 100.0;
                let elapsed = training_start_time.elapsed().as_secs();
                let eta =
                    (elapsed as f32 / (epoch + 1) as f32 * (max_epochs - epoch - 1) as f32) as u64;

                println!(
                    "[{:3}/{}] {:6.1}% | Loss: {:.4} | PPL: {:6.2} | LR: {:.6} | Grad: {:6.4} | Speed: {:5.1} samples/s | ETA: {}s",
                    epoch + 1,
                    max_epochs,
                    progress,
                    avg_loss,
                    perplexity,
                    current_lr,
                    avg_grad_norm,
                    samples_per_sec,
                    eta
                );
            }

            // 🔥 优化3：早停机制检查
            if early_stopping.should_stop(avg_loss, epoch) {
                let (best_loss, best_epoch) = early_stopping.best_state();
                println!("\n🛑 早停触发:");
                println!("   • 最佳epoch: {}", best_epoch + 1);
                println!("   • 最佳loss: {:.4}", best_loss);
                println!("   • 停止epoch: {}", epoch + 1);
                println!("   • 节省时间: {} epochs", max_epochs - epoch);

                self.set_training_mode(false);
                return epoch + 1; // 返回实际训练的epoch数
            }
        }

        self.set_training_mode(false);
        max_epochs
    }

    /// Add tokens to the context window, maintaining the maximum length
    pub fn add_to_context(&mut self, tokens: &[usize]) {
        // Add new tokens to the context window
        self.context_window.extend_from_slice(tokens);

        // If context exceeds maximum length, remove oldest tokens
        if self.context_window.len() > self.max_context_length {
            let excess = self.context_window.len() - self.max_context_length;
            self.context_window.drain(0..excess);
        }
    }

    /// Clear the context window
    pub fn clear_context(&mut self) {
        self.context_window.clear();
    }

    /// 启用所有transformer层的KV缓存
    ///
    /// KV缓存可以显著加速推理速度（10-100倍），但不能用于训练。
    /// 适用场景：交互式对话生成、逐token生成等
    pub fn enable_kv_cache(&mut self) {
        for layer in &mut self.network {
            // 只对SelfAttention层启用KV缓存
            // TransformerBlock内部包含SelfAttention，需要特殊处理
            // 这里我们通过layer_type判断
            if layer.layer_type() == "TransformerBlock" {
                // TransformerBlock需要向下传递enable命令
                // 由于Layer trait没有enable_kv_cache方法，
                // 我们需要在TransformerBlock中单独实现
                // 暂时通过unsafe转换实现
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.enable_kv_cache();
                }
            }
        }
    }

    /// 禁用所有transformer层的KV缓存
    pub fn disable_kv_cache(&mut self) {
        for layer in &mut self.network {
            if layer.layer_type() == "TransformerBlock" {
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.disable_kv_cache();
                }
            }
        }
    }

    /// 清空所有transformer层的KV缓存（保持启用状态）
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.network {
            if layer.layer_type() == "TransformerBlock" {
                unsafe {
                    let ptr = layer.as_mut() as *mut dyn Layer
                        as *mut crate::transformer::TransformerBlock;
                    (*ptr).attention.clear_kv_cache();
                }
            }
        }
    }

    /// Get current context as token IDs
    #[allow(dead_code)]
    pub fn get_context(&self) -> &[usize] {
        &self.context_window
    }

    /// Set a fixed context
    #[allow(dead_code)]
    pub fn set_context(&mut self, tokens: Vec<usize>) {
        self.context_window = tokens;
        // Ensure context doesn't exceed maximum length
        if self.context_window.len() > self.max_context_length {
            let excess = self.context_window.len() - self.max_context_length;
            self.context_window.drain(0..excess);
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // 使用 vocab 模块中的全局 Jieba 实例
        // 不再每次调用都初始化 Jieba

        // Check if the text contains Chinese characters
        let has_chinese = text
            .chars()
            .any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);

        let mut tokens = Vec::new();

        if has_chinese {
            // 使用 vocab 的 encode_sequence 方法，它内部使用全局 Jieba 实例
            return self.vocab.encode_sequence(text);
        } else {
            // Use the original method for non-Chinese text
            for word in text.split_whitespace() {
                // Special case for end token
                if word == "</s>" {
                    if let Some(token_id) = self.vocab.encode(word) {
                        tokens.push(token_id);
                    }
                    continue;
                }

                let mut current_word = String::new();

                for c in word.chars() {
                    if c.is_ascii_punctuation() {
                        // If we have a word before the punctuation, add it
                        if !current_word.is_empty() {
                            if let Some(token_id) = self.vocab.encode(&current_word) {
                                tokens.push(token_id);
                            }
                            current_word.clear();
                        }

                        // Add the punctuation as its own token
                        if let Some(token_id) = self.vocab.encode(&c.to_string()) {
                            tokens.push(token_id);
                        }
                    } else {
                        current_word.push(c);
                    }
                }

                // Add any remaining word
                if !current_word.is_empty()
                    && let Some(token_id) = self.vocab.encode(&current_word)
                {
                    tokens.push(token_id);
                }
            }
        }

        tokens
    }

    fn apply_temperature(probs: &Array2<f32>, temperature: f32) -> Array2<f32> {
        if temperature <= 0.0 {
            return probs.clone();
        }

        let power = 1.0 / temperature;
        let mut adjusted = probs.clone();

        for mut row in adjusted.rows_mut() {
            let mut sum = 0.0;
            for value in row.iter_mut() {
                *value = (*value).max(SOFTMAX_EPSILON).powf(power);
                sum += *value;
            }

            if sum > 0.0 {
                for value in row.iter_mut() {
                    *value /= sum;
                }
            }
        }

        adjusted
    }

    fn greedy_decode(probs: &Array2<f32>) -> Vec<usize> {
        probs
            .map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .to_vec()
    }

    /// Top-k sampling: only consider the k most probable tokens
    fn top_k_sampling(&self, probs: &Array2<f32>, k: usize) -> Vec<usize> {
        let mut result = Vec::new();

        for row in probs.rows() {
            // Create a vector of (probability, index) pairs
            let mut prob_idx_pairs: Vec<(f32, usize)> = row
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (prob, idx))
                .collect();

            // Sort by probability in descending order and take top k
            prob_idx_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            let top_k_pairs = &prob_idx_pairs[..k.min(prob_idx_pairs.len())];

            // Create a new probability distribution from top-k tokens
            let mut top_k_probs = vec![0.0; self.vocab.words.len()];
            let mut sum = 0.0;

            for &(prob, idx) in top_k_pairs {
                top_k_probs[idx] = prob;
                sum += prob;
            }

            // Normalize the probabilities
            if sum > 0.0 {
                for p in &mut top_k_probs {
                    *p /= sum;
                }
            }

            // Sample from the top-k distribution
            result.push(self.sample_from_probs(&top_k_probs));
        }

        result
    }

    /// Top-p (nucleus) sampling: consider the smallest set of tokens whose cumulative probability exceeds p
    fn top_p_sampling(&self, probs: &Array2<f32>, p: f32) -> Vec<usize> {
        let mut result = Vec::new();

        for row in probs.rows() {
            // Create a vector of (probability, index) pairs and sort by probability in descending order
            let mut prob_idx_pairs: Vec<(f32, usize)> = row
                .iter()
                .enumerate()
                .map(|(idx, &prob)| (prob, idx))
                .collect();

            prob_idx_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

            // Find the smallest set of tokens whose cumulative probability exceeds p
            let mut cumulative_prob = 0.0;
            let mut selected_pairs = Vec::new();

            for &(prob, idx) in &prob_idx_pairs {
                cumulative_prob += prob;
                selected_pairs.push((prob, idx));

                if cumulative_prob >= p {
                    break;
                }
            }

            // Create a new probability distribution from selected tokens
            let mut selected_probs = vec![0.0; self.vocab.words.len()];
            let mut sum = 0.0;

            for &(prob, idx) in &selected_pairs {
                selected_probs[idx] = prob;
                sum += prob;
            }

            // Normalize the probabilities
            if sum > 0.0 {
                for p in &mut selected_probs {
                    *p /= sum;
                }
            }

            // Sample from the selected distribution
            result.push(self.sample_from_probs(&selected_probs));
        }

        result
    }

    /// Sample from a probability distribution
    fn sample_from_probs(&self, probs: &[f32]) -> usize {
        let mut rng = rng();
        let sum: f32 = probs.iter().sum();

        if sum == 0.0 {
            return rng.random_range(0..probs.len());
        }

        let mut normalized_probs = Vec::new();
        let mut cumsum = 0.0;

        for &prob in probs {
            cumsum += prob / sum;
            normalized_probs.push(cumsum);
        }

        let rand_val: f32 = rng.random();
        for (i, &cum_prob) in normalized_probs.iter().enumerate() {
            if rand_val <= cum_prob {
                return i;
            }
        }

        probs.len() - 1
    }

    /// Beam search implementation
    fn beam_search(&mut self, text: &str, beam_width: usize, max_length: usize) -> String {
        // Tokenize the input text
        let initial_tokens = self.tokenize(text);
        if initial_tokens.is_empty() {
            return String::new();
        }

        // Initialize beam with the initial sequence
        let mut current_beams = vec![(initial_tokens.clone(), 0.0f32)]; // (sequence, log_probability)

        for _ in initial_tokens.len()..max_length {
            let mut candidates = Vec::new();

            for (seq, log_prob) in &current_beams {
                // Get model prediction for the current sequence
                let input =
                    Array2::from_shape_vec((1, seq.len()), seq.iter().map(|&x| x as f32).collect())
                        .unwrap();

                let mut input_tensor = input;
                for layer in &mut self.network {
                    input_tensor = layer.forward(&input_tensor);
                }

                let logits = input_tensor;
                let probs = softmax(&logits);

                // Get the probabilities for the last token position
                let last_token_probs = probs.row(probs.nrows() - 1);

                // Get top-k candidates for this sequence
                let mut token_probs: Vec<(f32, usize)> = last_token_probs
                    .iter()
                    .enumerate()
                    .map(|(idx, &prob)| (prob, idx))
                    .collect();

                token_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));

                // Add candidates with updated sequences and log probabilities
                for &(prob, token_id) in token_probs.iter().take(beam_width) {
                    if prob > 0.0 {
                        // Only consider non-zero probability tokens
                        let mut new_seq = seq.clone();
                        new_seq.push(token_id);
                        let new_log_prob = log_prob + prob.ln(); // Add log probabilities
                        candidates.push((new_seq, new_log_prob));
                    }
                }
            }

            // Sort candidates by log probability and keep the top beam_width
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            current_beams = candidates.into_iter().take(beam_width).collect();

            // Check if any beam has generated an end token
            if current_beams.iter().any(|(seq, _)| {
                if let Some(&last_token) = seq.last() {
                    self.vocab.decode.get(&last_token) == Some(&"</s>".to_string())
                } else {
                    false
                }
            }) {
                break;
            }
        }

        // Select the beam with the highest probability
        if let Some((best_seq, _)) = current_beams
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        {
            // Convert token IDs back to text
            let token_strs = best_seq
                .iter()
                .map(|&t| self.vocab.decode[&t].clone())
                .filter(|s| s != "</s>") // Remove end token
                .collect::<Vec<String>>();

            // Apply post-processing to improve fluency and accuracy
            let raw_output = token_strs.join(" ");
            self.post_process_chinese_text(&raw_output)
        } else {
            String::new() // Return empty string if no valid sequence is found
        }
    }

    /// Generate text based on the given context tokens
    fn generate_with_context(
        &mut self,
        context_tokens: &[usize],
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> String {
        let mut tokenized = context_tokens.to_vec();
        let mut output_tokens: Vec<usize> = Vec::new();

        if tokenized.is_empty() {
            return String::new();
        }

        let input_len = tokenized.len();

        if input_len >= MAX_SEQ_LEN {
            return String::new();
        }

        let max_new_tokens = 20.min(MAX_SEQ_LEN - input_len);

        for _ in 0..max_new_tokens {
            let token_input = Array2::from_shape_vec(
                (1, tokenized.len()),
                tokenized.iter().map(|&x| x as f32).collect(),
            )
            .unwrap();
            let mut input = token_input;

            for layer in &mut self.network {
                input = layer.forward(&input);
            }

            let logits = input;

            if logits.shape()[0] == 0 {
                break;
            }

            let last_logit = logits
                .row(logits.shape()[0] - 1)
                .to_owned()
                .insert_axis(Axis(0));

            let probs = softmax(&last_logit);

            let adjusted_probs = Self::apply_temperature(&probs, temperature);

            let tokens = if top_k > 0 {
                self.top_k_sampling(&adjusted_probs, top_k)
            } else {
                self.top_p_sampling(&adjusted_probs, top_p)
            };

            let next_token = tokens[tokens.len() - 1];

            output_tokens.push(next_token);
            tokenized.push(next_token);

            if next_token == self.vocab.encode("</s>").unwrap() {
                break;
            }
        }

        let token_strs = output_tokens
            .iter()
            .map(|t| self.vocab.decode[t].clone())
            .collect::<Vec<String>>();

        let raw_output = token_strs.join(" ");
        self.post_process_chinese_text(&raw_output)
    }

    /// Post-process generated Chinese text to improve fluency and accuracy
    pub fn post_process_chinese_text(&self, text: &str) -> String {
        // Remove extra spaces between Chinese characters
        let mut result = String::new();
        let mut chars = text.chars().peekable();

        while let Some(ch) = chars.next() {
            result.push(ch);

            // If current and next characters are both Chinese, don't add space
            if let Some(&next_ch) = chars.peek() {
                if self.is_chinese_char(ch) && self.is_chinese_char(next_ch) {
                    // Skip any space between Chinese characters
                    if next_ch == ' ' {
                        chars.next(); // consume the space
                    }
                }
            }
        }

        // Additional processing could include:
        // - Grammar pattern correction
        // - Ensuring proper sentence structure
        // - Removing repetitive patterns

        result
    }

    /// Helper function to check if a character is Chinese
    fn is_chinese_char(&self, ch: char) -> bool {
        (ch as u32) >= 0x4E00 && (ch as u32) <= 0x9FFF
    }

    fn cross_entropy_loss_step(probs: &Array2<f32>, target: &[usize]) -> f32 {
        let mut loss = 0.0;
        let n_targets = target.len() as f32;

        for (row_idx, &target_idx) in target.iter().enumerate() {
            let prob_target = probs[[row_idx, target_idx]]; // Get probability of correct token
            loss -= prob_target.max(LOG_EPSILON).ln(); // 使用统一的LOG_EPSILON保证数值稳定性
        }

        loss / n_targets
    }

    fn compute_gradients_step(probs: &Array2<f32>, target: &[usize]) -> Array2<f32> {
        let mut grads = probs.clone(); // Start with softmax probabilities

        if probs.shape()[0] != target.len() {
            panic!("Probs and target must have the same number of rows");
        }

        let batch_size = target.len() as f32;

        // Compute correct softmax + cross-entropy gradient: softmax - one_hot(target)
        for (row_idx, &target_idx) in target.iter().enumerate() {
            grads[[row_idx, target_idx]] -= 1.0; // Convert to: p - y (where y is one-hot)
        }

        // Normalize by batch size for stable training
        grads.mapv_inplace(|x| x / batch_size);

        grads
    }

    fn clip_gradients(grads: &mut Array2<f32>, max_norm: f32) {
        // Calculate L2 norm of gradients
        let norm = grads.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // If norm exceeds max_norm, scale gradients down
        if norm > max_norm {
            let scale = max_norm / norm;
            grads.mapv_inplace(|x| x * scale);
        }
    }
}
