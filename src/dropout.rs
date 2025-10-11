use ndarray::Array2;
use rand::Rng;

use crate::{adam::Adam, llm::Layer};

pub struct Dropout {
    pub dropout_rate: f32,
    mask: Option<Array2<f32>>, // Cached mask for backward pass
    training: bool, // Whether in training or inference mode
}

impl Dropout {
    pub fn new(dropout_rate: f32) -> Self {
        Self {
            dropout_rate,
            mask: None,
            training: true, // Default to training mode
        }
    }

    pub fn set_training_mode(&mut self, training: bool) {
        self.training = training;
    }

    fn create_mask(&self, shape: (usize, usize)) -> Array2<f32> {
        let mut rng = rand::thread_rng();
        let (rows, cols) = shape;
        let mut mask = Array2::zeros((rows, cols));
        
        for mut row in mask.rows_mut() {
            for element in row.iter_mut() {
                // Generate a random number between 0 and 1
                let random_val: f32 = rng.random();
                
                // If random_val > dropout_rate, keep the neuron (set to 1), else drop it (set to 0)
                *element = if random_val > self.dropout_rate { 1.0 } else { 0.0 };
            }
        }
        
        mask
    }
}

impl Layer for Dropout {
    fn layer_type(&self) -> &str {
        "Dropout"
    }

    fn forward(&mut self, input: &Array2<f32>) -> Array2<f32> {
        if self.training && self.dropout_rate > 0.0 {
            let mask = self.create_mask(input.dim());
            self.mask = Some(mask.clone());
            
            // Apply dropout: multiply input by mask and scale by (1 / (1 - dropout_rate))
            // Scaling is important to maintain the expected value during training
            let scale_factor = 1.0 / (1.0 - self.dropout_rate);
            let mut result = input.clone();
            result *= &mask;  // Element-wise multiplication
            result *= scale_factor;  // Scale by factor
            result
        } else {
            // In inference mode or when dropout_rate is 0, return input unchanged
            input.clone()
        }
    }

    fn backward(&mut self, grads: &Array2<f32>, _lr: f32) -> Array2<f32> {
        if self.training && self.dropout_rate > 0.0 {
            // Apply the same mask that was used in the forward pass to the gradients
            let mask = self.mask.as_ref().unwrap();
            let scale_factor = 1.0 / (1.0 - self.dropout_rate);
            let mut result = grads.clone();
            result *= mask;  // Element-wise multiplication
            result *= scale_factor;  // Scale by factor
            result
        } else {
            // Return gradients unchanged if not in training mode or dropout_rate is 0
            grads.clone()
        }
    }

    fn parameters(&self) -> usize {
        // Dropout layer has no trainable parameters
        0
    }
}