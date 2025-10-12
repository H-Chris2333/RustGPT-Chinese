use ndarray::Array2;

use crate::{EMBEDDING_DIM, MAX_SEQ_LEN};

pub struct PositionEncoding {
    pub encoding: Array2<f32>,
}

impl PositionEncoding {
    pub fn new() -> Self {
        // Initialize the position encoding matrix (MAX_SEQ_LEN x EMBEDDING_DIM)
        let mut encoding = Array2::zeros((MAX_SEQ_LEN, EMBEDDING_DIM));
        
        for pos in 0..MAX_SEQ_LEN {
            for i in 0..EMBEDDING_DIM {
                let angle = pos as f32 / 10000f32.powf((i / 2) as f32 / EMBEDDING_DIM as f32);
                
                if i % 2 == 0 {
                    encoding[[pos, i]] = angle.sin();
                } else {
                    encoding[[pos, i]] = angle.cos();
                }
            }
        }
        
        Self { encoding }
    }

    pub fn get_encoding(&self, position: usize, dimension: usize) -> f32 {
        if position >= MAX_SEQ_LEN || dimension >= EMBEDDING_DIM {
            panic!("Position or dimension out of bounds");
        }
        self.encoding[[position, dimension]]
    }
    
    /// Apply position encoding to input embeddings
    /// input: (seq_len, embedding_dim)
    #[allow(dead_code)]
    pub fn apply_to_input(&self, input: &mut Array2<f32>) {
        let (seq_len, embedding_dim) = input.dim();
        
        // Determine how many positions we can encode based on input length
        let positions_to_encode = std::cmp::min(seq_len, MAX_SEQ_LEN);
        let dims_to_encode = std::cmp::min(embedding_dim, EMBEDDING_DIM);
        
        for pos in 0..positions_to_encode {
            for dim in 0..dims_to_encode {
                input[[pos, dim]] += self.encoding[[pos, dim]];
            }
        }
    }
}

// For Chinese language, we might also want to implement relative position encoding
// which works better with the structure of Chinese text

#[allow(dead_code)]
pub struct RelativePositionEncoding {
    pub encoding: Array2<f32>,
    pub max_offset: usize,
}

impl RelativePositionEncoding {
    #[allow(dead_code)]
    pub fn new(max_offset: usize) -> Self {
        // Create relative position encoding matrix
        let total_positions = 2 * max_offset + 1;
        let mut encoding = Array2::zeros((total_positions, EMBEDDING_DIM));
        
        for offset in 0..total_positions {
            let relative_pos = offset as i32 - max_offset as i32;
            for i in 0..EMBEDDING_DIM {
                let angle = relative_pos as f32 / 10000f32.powf((i / 2) as f32 / EMBEDDING_DIM as f32);
                
                if i % 2 == 0 {
                    encoding[[offset, i]] = angle.sin();
                } else {
                    encoding[[offset, i]] = angle.cos();
                }
            }
        }
        
        Self { encoding, max_offset }
    }

    #[allow(dead_code)]
    /// Get encoding for a relative position
    pub fn get_encoding(&self, relative_pos: i32) -> Option<Vec<f32>> {
        let index = relative_pos as i32 + self.max_offset as i32;
        if index < 0 || index >= (2 * self.max_offset + 1) as i32 {
            return None;
        }
        let index = index as usize;
        Some(self.encoding.row(index).to_vec())
    }
}