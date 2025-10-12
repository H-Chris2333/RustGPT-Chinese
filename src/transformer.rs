use ndarray::Array2;

use crate::{
    dropout::Dropout, feed_forward::FeedForward, layer_norm::LayerNorm, llm::Layer, self_attention::SelfAttention,
};
pub struct TransformerBlock {
    attention: SelfAttention,
    feed_forward: FeedForward,
    dropout1: Dropout, // After attention
    dropout2: Dropout, // After feed forward
    norm1: LayerNorm, // After attention
    norm2: LayerNorm, // After feed forward
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
        let attention_out = self.attention.forward(input);
        let norm1_out = self.norm1.normalize(&attention_out);
        let dropout1_out = self.dropout1.forward(&norm1_out);

        let feed_forward_out = self.feed_forward.forward(&dropout1_out);
        let dropout2_out = self.dropout2.forward(&feed_forward_out);

        self.norm2.normalize(&dropout2_out)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let grad_norm2 = self.norm2.backward(grads, lr);

        let grad_dropout2 = self.dropout2.backward(&grad_norm2, lr);

        let grad_ffn = self.feed_forward.backward(&grad_dropout2, lr);

        let grad_dropout1 = self.dropout1.backward(&grad_ffn, lr);

        let grad_norm1 = self.norm1.backward(&grad_dropout1, lr);

        self.attention.backward(&grad_norm1, lr)
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
