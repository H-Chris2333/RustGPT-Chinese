use ndarray::{Array2, s};
use rand_distr::{Distribution, Normal};

use crate::{EMBEDDING_DIM, MAX_SEQ_LEN, adam::Adam, llm::Layer, position_encoding::PositionEncoding, semantic_enhancer::SemanticEnhancer, vocab::Vocab};

pub struct Embeddings {
    pub token_embeddings: Array2<f32>,
    pub position_encoder: PositionEncoding,
    pub semantic_enhancer: SemanticEnhancer,
    pub cached_input: Option<Array2<f32>>,
    pub token_optimizer: Adam,
}

impl Default for Embeddings {
    fn default() -> Self {
        let vocab = Vocab::default();
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), EMBEDDING_DIM),
            position_encoder: PositionEncoding::new(),
            semantic_enhancer: SemanticEnhancer::new(vocab.clone()),
            cached_input: None,
            token_optimizer: Adam::new((vocab.words.len(), EMBEDDING_DIM)),
        }
    }
}

impl Embeddings {
    pub fn new(vocab: Vocab) -> Self {
        Self {
            token_embeddings: Self::init_embeddings(vocab.words.len(), EMBEDDING_DIM),
            position_encoder: PositionEncoding::new(),
            semantic_enhancer: SemanticEnhancer::new(vocab.clone()),
            cached_input: None,
            token_optimizer: Adam::new((vocab.words.len(), EMBEDDING_DIM)),
        }
    }

    fn init_embeddings(vocab_size: usize, embedding_dim: usize) -> Array2<f32> {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 0.02).unwrap(); // Increased for better learning
        Array2::from_shape_fn((vocab_size, embedding_dim), |_| normal.sample(&mut rng))
    }

    fn get_token_embeddings(embeddings: &Array2<f32>, token_ids: &[usize]) -> Array2<f32> {
        let mut token_embeds = Array2::<f32>::zeros((token_ids.len(), embeddings.ncols()));
        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= embeddings.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    embeddings.nrows()
                );
            }
            token_embeds.row_mut(i).assign(&embeddings.row(token_id));
        }
        token_embeds
    }

    pub fn embed_tokens(&self, token_ids: &[usize]) -> Array2<f32> {
        let token_embeds = Self::get_token_embeddings(&self.token_embeddings, token_ids);
        let mut position_embeds = Array2::<f32>::zeros((token_ids.len(), EMBEDDING_DIM));
        
        // Apply positional encoding to each position
        for (i, _) in token_ids.iter().enumerate() {
            for j in 0..EMBEDDING_DIM {
                position_embeds[[i, j]] = self.position_encoder.get_encoding(i, j);
            }
        }
        
        token_embeds + position_embeds // Element-wise sum
    }
}

impl Layer for Embeddings {
    fn layer_type(&self) -> &str {
        "Embeddings"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        // input shape is [1, sequence_length]
        self.cached_input = Some(input.clone());
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let embeddings = self.embed_tokens(&token_ids); // shape is [sequence_length, embedding_dim]
        
        // Apply semantic enhancement
        self.semantic_enhancer.enhance_embeddings(&embeddings, &token_ids)
    }

    fn backward(&mut self, grads: &Array2<f32>, lr: f32) -> Array2<f32> {
        let input = self.cached_input.as_ref().unwrap();
        let token_ids: Vec<usize> = input.iter().map(|&x| x as usize).collect();
        let grads = grads.view(); // (sequence_length, embedding_dim)

        // Initialize gradients for token embeddings only (positional encoding is fixed)
        let mut token_grads = Array2::zeros(self.token_embeddings.dim());

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.token_embeddings.nrows() {
                panic!(
                    "Token ID {} out of bounds for vocab size {}",
                    token_id,
                    self.token_embeddings.nrows()
                );
            }
            let grad_row = grads.row(i);

            // Accumulate token embedding gradients
            {
                let mut token_row = token_grads.row_mut(token_id);
                token_row += &grad_row;
            }
        }

        self.token_optimizer
            .step(&mut self.token_embeddings, &token_grads, lr);

        // Return gradient to propagate further back
        grads.to_owned()
    }

    fn parameters(&self) -> usize {
        self.token_embeddings.len()  // Only token embeddings are trainable, positional encoding is fixed
    }
}
