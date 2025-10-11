use std::collections::{HashMap, HashSet};

use bincode::Encode;
use jieba_rs::Jieba;
use regex::Regex;

#[derive(Clone, Encode)]
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    pub decode: HashMap<usize, String>,
    pub words: Vec<String>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new(Self::default_words())
    }
}

impl Vocab {
    pub fn new(words: Vec<&str>) -> Self {
        let mut encode = HashMap::new();
        let mut decode = HashMap::new();

        for (i, &word) in words.iter().enumerate() {
            println!("Adding word: {word} to encoding: {i}");
            encode.insert(word.to_string(), i);
            decode.insert(i, word.to_string());
        }

        Vocab {
            encode,
            decode,
            words: words.iter().map(|w| w.to_string()).collect(),
        }
    }

    /// Convert a word to its token index
    pub fn encode(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied()
    }

    /// Convert a token index back to a word
    #[allow(dead_code)]
    pub fn decode(&self, token_id: usize) -> Option<&String> {
        self.decode.get(&token_id)
    }

    pub fn default_words() -> Vec<&'static str> {
        vec!["hello", "world", "this", "is", "rust", "</s>"]
    }

    /// Process text data to extract vocabulary words and add them to the vocabulary set
    pub fn process_text_for_vocab(texts: &[String], vocab_set: &mut HashSet<String>) {
        // Add end of sequence token
        vocab_set.insert("</s>".to_string());
        
        // Initialize Jieba tokenizer
        let jieba = Jieba::new();

        // Process all training examples for vocabulary
        for text in texts {
            // Check if the text contains Chinese characters
            let has_chinese = text.chars().any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);
            
            if has_chinese {
                // Use Jieba for Chinese text tokenization
                let tokens = jieba.cut(text, false);
                for token in tokens {
                    if !token.trim().is_empty() {
                        vocab_set.insert(token.trim().to_string());
                    }
                }
                
                // Process common Chinese idioms and phrases that might be missed by Jieba
                Self::extract_chinese_phrases(text, vocab_set);
            } else {
                // Use the original method for non-Chinese text
                for word in text.split_whitespace() {
                    // Handle punctuation by splitting it from words
                    let mut current = String::new();
                    for c in word.chars() {
                        if c.is_ascii_punctuation() {
                            if !current.is_empty() {
                                vocab_set.insert(current.clone());
                                current.clear();
                            }
                            vocab_set.insert(c.to_string());
                        } else {
                            current.push(c);
                        }
                    }
                    if !current.is_empty() {
                        vocab_set.insert(current);
                    }
                }
            }
        }
    }
    
    /// Extract common Chinese phrases and idioms that might be missed by simple tokenization
    fn extract_chinese_phrases(text: &str, vocab_set: &mut HashSet<String>) {
        // Common Chinese idioms (四字成语) - these are often not segmented properly by Jieba
        let idiom_regex = Regex::new(r"[\u4e00-\u9fff]{4}").unwrap();
        for mat in idiom_regex.find_iter(text) {
            let idiom = mat.as_str();
            if Self::is_common_chinese_idiom(idiom) {
                vocab_set.insert(idiom.to_string());
            }
        }
        
        // Common multi-character phrases that might be relevant
        let phrase_regex = Regex::new(r"[\u4e00-\u9fff]{2,6}").unwrap();
        for mat in phrase_regex.find_iter(text) {
            let phrase = mat.as_str();
            if Self::is_meaningful_phrase(phrase) {
                vocab_set.insert(phrase.to_string());
            }
        }
    }
    
    /// Check if a 4-character string is a common Chinese idiom
    /// This is a simplified check - in practice, we'd use a comprehensive idiom dictionary
    fn is_common_chinese_idiom(idiom: &str) -> bool {
        // In a real implementation, we would check against a dictionary of idioms
        // For now, just checking if all characters are Chinese
        idiom.chars().all(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF)
    }
    
    /// Check if a multi-character string is likely a meaningful phrase
    fn is_meaningful_phrase(phrase: &str) -> bool {
        // In a real implementation, we would check against a dictionary or use other heuristics
        // For now, just checking if all characters are Chinese and length is reasonable
        phrase.chars().all(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF) && phrase.len() >= 2
    }
}

// Helper method to check if a character is Chinese
trait IsChinese {
    fn is_chinese(&self) -> bool;
}

impl IsChinese for char {
    fn is_chinese(&self) -> bool {
        (*self as u32) >= 0x4E00 && (*self as u32) <= 0x9FFF
    }
}

impl From<Vocab> for String {
    fn from(val: Vocab) -> Self {
        String::from_iter(
            val.words
                .iter()
                .enumerate()
                .map(|(i, str)| format!("({i},{str}),")),
        )
    }
}
