use llm::Vocab;

#[test]
fn test_vocab_encode_decode() {
    let words = vec!["hello", "world", "this", "is", "rust"];
    let vocab = Vocab::new(words);

    // Test encoding - with special tokens, "hello" should be at index 7
    assert_eq!(vocab.encode("hello"), Some(7));
    assert_eq!(vocab.encode("world"), Some(8));
    assert_eq!(vocab.encode("unknown"), None);

    // Test decoding
    assert_eq!(vocab.decode(7).map(|s| s.as_str()), Some("hello"));
    assert_eq!(vocab.decode(8).map(|s| s.as_str()), Some("world"));
    assert_eq!(vocab.decode(999), None);
}

#[test]
fn test_vocab_default() {
    let vocab = Vocab::default();

    // Test that default vocab contains expected words - they should be at higher indices due to special tokens
    assert!(vocab.encode("hello").is_some());
    assert!(vocab.encode("world").is_some());
    assert!(vocab.encode("</s>").is_some()); // This is a special token at index 3
    
    // Verify special token IDs
    assert_eq!(vocab.eos_token_id(), 3); // </s> is at index 3
    assert_eq!(vocab.pad_token_id(), 0); // <|pad|> is at index 0
    assert_eq!(vocab.unk_token_id(), 1); // <|unk|> is at index 1
}

#[test]
fn test_vocab_basic_operations() {
    let vocab = Vocab::new(vec!["hello", "world", "test"]);
    
    // With special tokens, "hello" should be at index 7 (0-6 are special tokens)
    assert_eq!(vocab.encode("hello"), Some(7));
    assert_eq!(vocab.decode(7), Some(&"hello".to_string()));
    assert_eq!(vocab.len(), 10); // 3 words + 7 special tokens
}

#[test]
fn test_vocab_special_tokens() {
    let special_tokens = Vocab::default_special_tokens();
    assert!(special_tokens.contains_key("<|pad|>"));
    assert!(special_tokens.contains_key("<|unk|>"));
    assert!(special_tokens.contains_key("</s>"));
}

#[test]
fn test_encode_decode_sequence() {
    let vocab = Vocab::new(vec!["hello", "world", "rust"]);
    
    let text = "hello world";
    let encoded = vocab.encode_sequence(text);
    let decoded = vocab.decode_sequence(&encoded);
    
    assert!(!encoded.is_empty());
    assert!(decoded.contains("hello"));
    assert!(decoded.contains("world"));
}

#[test]
fn test_build_from_texts() {
    let texts = vec!["hello world".to_string(), "world of rust".to_string()];
    let vocab = Vocab::build_from_texts(&texts);
    
    assert!(vocab.encode("hello").is_some());
    assert!(vocab.encode("world").is_some());
    assert!(vocab.encode("rust").is_some());
}
