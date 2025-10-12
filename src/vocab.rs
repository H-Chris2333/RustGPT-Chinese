use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;

use bincode::{Encode, Decode};
use jieba_rs::Jieba;
use regex::Regex;
use serde::{Serialize, Deserialize};

use std::fs;

static COMMON_IDIOM_SET: OnceLock<HashSet<String>> = OnceLock::new();

fn common_idioms() -> &'static HashSet<String> {
    COMMON_IDIOM_SET.get_or_init(|| {
        load_common_idioms_from_file()
            .expect("Failed to load chinese idioms from data/chinese_idioms.json")
    })
}

fn load_common_idioms_from_file() -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let idioms_file_path = "data/chinese_idioms.json";
    let idioms_json = fs::read_to_string(idioms_file_path)?;
    let idioms: Vec<String> = serde_json::from_str(&idioms_json)?;
    Ok(HashSet::from_iter(idioms))
}

#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Vocab {
    pub encode: HashMap<String, usize>,
    pub decode: HashMap<usize, String>,
    pub words: Vec<String>,
    pub special_tokens: HashMap<String, usize>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new_with_special_tokens(Self::default_words(), Self::default_special_tokens())
    }
}

impl Vocab {
    /// Create a new vocabulary from a list of words
    pub fn new(words: Vec<&str>) -> Self {
        Self::new_with_special_tokens(words, Self::default_special_tokens())
    }

    /// Create a new vocabulary with special tokens
    pub fn new_with_special_tokens(words: Vec<&str>, special_tokens: HashMap<String, usize>) -> Self {
        let mut encode = HashMap::new();
        let mut decode = HashMap::new();

        // Add special tokens first with their predefined IDs
        println!("Adding special tokens to vocabulary:");
        for (token, id) in &special_tokens {
            encode.insert(token.clone(), *id);
            decode.insert(*id, token.clone());
            println!("  Added special token: '{}' with ID {}", token, id);
        }

        // Add remaining words starting from the next available ID
        println!("Adding regular vocabulary words:");
        let mut next_id = special_tokens.values().max().unwrap_or(&0) + 1;
        for word in words.iter() {
            let word_str = word.to_string();
            if !encode.contains_key(&word_str) {
                encode.insert(word_str.clone(), next_id);
                decode.insert(next_id, word_str.clone());
                println!("  Added word: '{}' with ID {}", word_str, next_id);
                next_id += 1;
            } else {
                // If the word already exists (e.g., as a special token), skip it
                println!("  Skipped duplicate word: '{}' (already exists with ID {})", word_str, encode[&word_str]);
            }
        }

        let all_words: Vec<String> = decode.values().cloned().collect();

        Vocab {
            encode,
            decode,
            words: all_words,
            special_tokens,
        }
    }

    /// Convert a word to its token index
    pub fn encode(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied()
    }

    /// Convert a word to its token index, with fallback to unknown token
    pub fn encode_with_unk(&self, word: &str) -> usize {
        match self.encode(word) {
            Some(id) => id,
            None => *self.special_tokens.get("<|unk|>").unwrap_or(&0),
        }
    }

    /// Convert a token index back to a word
    #[allow(dead_code)]
    pub fn decode(&self, token_id: usize) -> Option<&String> {
        self.decode.get(&token_id)
    }

    /// Encode a sequence of text into token IDs
    pub fn encode_sequence(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();
        
        // Check if the text contains Chinese characters
        let has_chinese = text.chars().any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);
        
        if has_chinese {
            let jieba = Jieba::new();
            let seg_list = jieba.cut(text, false);
            for word in seg_list {
                if !word.trim().is_empty() {
                    let token_id = self.encode_with_unk(word.trim());
                    tokens.push(token_id);
                }
            }
        } else {
            // For non-Chinese text, use simple tokenization
            for word in text.split_whitespace() {
                let token_id = self.encode_with_unk(word);
                tokens.push(token_id);
            }
        }
        
        tokens
    }

    /// Decode a sequence of token IDs back to text
    pub fn decode_sequence(&self, token_ids: &[usize]) -> String {
        let mut result = Vec::new();
        for &token_id in token_ids {
            if let Some(word) = self.decode.get(&token_id) {
                result.push(word.clone());
            }
        }
        result.join(" ")
    }

    /// Get the size of the vocabulary
    pub fn len(&self) -> usize {
        self.encode.len()
    }

    /// Check if the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.encode.is_empty()
    }

    /// Get the ID of the unknown token
    pub fn unk_token_id(&self) -> usize {
        *self.special_tokens.get("<|unk|>").unwrap_or(&0)
    }

    /// Get the ID of the padding token
    pub fn pad_token_id(&self) -> usize {
        *self.special_tokens.get("<|pad|>").unwrap_or(&0)
    }

    /// Get the ID of the end of sequence token
    pub fn eos_token_id(&self) -> usize {
        *self.special_tokens.get("</s>").unwrap_or(&0)
    }

    /// Get the ID of the start of sequence token
    pub fn bos_token_id(&self) -> usize {
        *self.special_tokens.get("<|bos|>").unwrap_or(&0)
    }

    pub fn default_words() -> Vec<&'static str> {
        vec!["hello", "world", "this", "is", "rust", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through", "during", "before", "after", "above", "below", "between", "among", "as", "if", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should", "now"]
    }

    pub fn default_special_tokens() -> HashMap<String, usize> {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|pad|>".to_string(), 0);
        special_tokens.insert("<|unk|>".to_string(), 1);
        special_tokens.insert("<|bos|>".to_string(), 2);
        special_tokens.insert("</s>".to_string(), 3);  // End of sequence
        special_tokens.insert("<|sep|>".to_string(), 4);  // Separator
        special_tokens.insert("<|cls|>".to_string(), 5);  // Classification
        special_tokens.insert("<|mask|>".to_string(), 6);  // Masked token
        special_tokens
    }

    /// Process text data to extract vocabulary words and add them to the vocabulary set
    pub fn process_text_for_vocab(texts: &[String], vocab_set: &mut HashSet<String>) {
        let mut vocab_log = String::new();
        let mut idiom_writer = std::io::BufWriter::new(std::io::sink());

        vocab_set.insert("<|pad|>".to_string());
        vocab_set.insert("<|unk|>".to_string());
        vocab_set.insert("<|bos|>".to_string());
        vocab_set.insert("</s>".to_string());
        vocab_set.insert("<|sep|>".to_string());
        vocab_set.insert("<|cls|>".to_string());
        vocab_set.insert("<|mask|>".to_string());

        vocab_log.push_str("Initialized special tokens.\n");

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
                        let token_trimmed = token.trim().to_string();
                        vocab_log.push_str(&format!("Token: {}\n", token_trimmed));
                        vocab_set.insert(token_trimmed);
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
                                vocab_log.push_str(&format!("Word: {}\n", current));
                                vocab_set.insert(current.clone());
                                current.clear();
                            }
                            vocab_log.push_str(&format!("Punctuation: {}\n", c));
                            vocab_set.insert(c.to_string());
                        } else {
                            current.push(c);
                        }
                    }
                    if !current.is_empty() {
                        vocab_log.push_str(&format!("Word: {}\n", current));
                        vocab_set.insert(current);
                    }
                }
            }
        }

        let _ = writeln!(idiom_writer, "{}", vocab_log);
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
        let mut chars = idiom.chars();
        if chars.clone().count() != 4 {
            return false;
        }
        if !chars.all(|c| c.is_chinese()) {
            return false;
        }
        common_idioms().contains(idiom)  // This works because HashSet<String> implements contains for &str
    }
    
    /// Check if a multi-character string is likely a meaningful phrase
    fn is_meaningful_phrase(phrase: &str) -> bool {
        let length = phrase.chars().count();
        if length < 2 || length > 8 {
            return false;
        }
        if !phrase.chars().all(|c| c.is_chinese()) {
            return false;
        }
        if length == 4 && Self::is_common_chinese_idiom(phrase) {
            return true;
        }
        let jieba = Jieba::new();
        let tokens = jieba.cut(phrase, false);
        if tokens.is_empty() {
            return false;
        }
        if tokens.len() == 1 {
            return true;
        }
        let total_len: usize = tokens.iter().map(|token| token.chars().count()).sum();
        total_len == length && tokens.len() <= 2
    }

    /// Build vocabulary from text files
    pub fn build_from_texts(texts: &[String]) -> Self {
        let mut vocab_set = HashSet::new();
        Self::process_text_for_vocab(texts, &mut vocab_set);
        
        let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
        vocab_words.sort(); // Sort for deterministic ordering
        
        // Create vectors of string references for the constructor
        let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
        let special_tokens = Self::default_special_tokens();
        
        Self::new_with_special_tokens(vocab_words_refs, special_tokens)
    }

    /// Save vocabulary to a file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        Ok(())
    }

    /// Load vocabulary from a file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let vocab: Vocab = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(vocab)
    }

    /// Add a word to the vocabulary
    pub fn add_word(&mut self, word: String) -> usize {
        if let Some(existing_id) = self.encode.get(&word) {
            return *existing_id;
        }

        let new_id = self.encode.len();
        self.encode.insert(word.clone(), new_id);
        self.decode.insert(new_id, word.clone());
        self.words.push(word);
        new_id
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
