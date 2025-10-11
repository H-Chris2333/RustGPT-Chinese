use llm::{position_encoding::PositionEncoding, MAX_SEQ_LEN, EMBEDDING_DIM};

#[test]
fn test_position_encoding_creation() {
    let pos_enc = PositionEncoding::new();
    assert_eq!(pos_enc.encoding.dim(), (MAX_SEQ_LEN, EMBEDDING_DIM));
}

#[test]
fn test_position_encoding_values() {
    let pos_enc = PositionEncoding::new();
    // Test some specific values
    let val1 = pos_enc.get_encoding(0, 0); // Should be sin(0) = 0
    let val2 = pos_enc.get_encoding(1, 0); // Should be sin(1/10000^0/512) = sin(1)
    
    assert!((val1 - 0.0).abs() < 1e-5); // Approximately 0
    assert!((val2 - 0.841471).abs() < 1e-5); // Approximately sin(1)
}

#[test]
fn test_apply_to_input() {
    let pos_enc = PositionEncoding::new();
    let mut input = ndarray::Array2::ones((5, EMBEDDING_DIM)); // 5 tokens, embedding_dim dimensions
    let original_sum = input.sum();
    
    pos_enc.apply_to_input(&mut input);
    
    // After applying position encoding, the sum should be different
    let new_sum = input.sum();
    assert_ne!(original_sum, new_sum);
}