use ndarray::Array2;

use crate::{
    dropout::Dropout, feed_forward::FeedForward, layer_norm::LayerNorm, llm::Layer, self_attention::SelfAttention,
};
pub struct TransformerBlock {
    pub attention: SelfAttention,
    pub feed_forward: FeedForward,
    pub dropout1: Dropout, // After attention
    pub dropout2: Dropout, // After feed forward
    pub norm1: LayerNorm, // After attention
    pub norm2: LayerNorm, // After feed forward
}

impl TransformerBlock {
    pub fn new(embedding_dim: usize, hidden_dim: usize) -> Self {
        TransformerBlock {
            attention: SelfAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, hidden_dim),
            dropout1: Dropout::new(0.1), // 10% dropout after attention
            dropout2: Dropout::new(0.1), // 10% dropout after feed forward
            norm1: LayerNorm::new(embedding_dim),
            norm2: LayerNorm::new(embedding_dim),
        }
    }
}

impl Layer for TransformerBlock {
    fn layer_type(&self) -> &str {
        "TransformerBlock"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // Pre-LN Transformer 架构（更稳定，GPT-2 及之后常用）

        // 第一个子层：Multi-Head Attention
        // x = input + dropout(attention(norm(input)))
        let norm1_out = self.norm1.normalize(input);
        let attention_out = self.attention.forward(&norm1_out);
        let dropout1_out = self.dropout1.forward(&attention_out);
        let x = input + &dropout1_out;  // 残差连接

        // 第二个子层：Feed-Forward Network
        // output = x + dropout(ffn(norm(x)))
        let norm2_out = self.norm2.normalize(&x);
        let feed_forward_out = self.feed_forward.forward(&norm2_out);
        let dropout2_out = self.dropout2.forward(&feed_forward_out);

        &x + &dropout2_out  // 残差连接
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        // 反向传播遵循前向传播的相反顺序

        // 第二个残差连接的梯度
        let grad_dropout2 = grads;
        let grad_x_from_residual2 = grads;

        let grad_dropout2_out = self.dropout2.backward(grad_dropout2, lr);
        let grad_ffn = self.feed_forward.backward(&grad_dropout2_out, lr);
        let grad_norm2 = self.norm2.backward(&grad_ffn, lr);

        // 累积来自第二个残差连接的梯度
        let grad_x = &grad_norm2 + grad_x_from_residual2;

        // 第一个残差连接的梯度
        let grad_dropout1 = &grad_x;
        let grad_input_from_residual1 = &grad_x;

        let grad_dropout1_out = self.dropout1.backward(grad_dropout1, lr);
        let grad_attention = self.attention.backward(&grad_dropout1_out, lr);
        let grad_norm1 = self.norm1.backward(&grad_attention, lr);

        // 累积来自第一个残差连接的梯度
        &grad_norm1 + grad_input_from_residual1
    }

    fn set_training_mode(&mut self, training: bool) {
        self.dropout1.set_training_mode(training);
        self.dropout2.set_training_mode(training);
        self.attention.set_training_mode(training);
        self.feed_forward.set_training_mode(training);
        self.norm1.set_training_mode(training);
        self.norm2.set_training_mode(training);
    }

    fn parameters(&self) -> usize {
        self.attention.parameters()
            + self.feed_forward.parameters()
            + self.dropout1.parameters()
            + self.dropout2.parameters()
            + self.norm1.parameters()
            + self.norm2.parameters()
    }
}
