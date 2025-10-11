use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::OnceLock;

use bincode::{Encode, Decode};
use jieba_rs::Jieba;
use regex::Regex;
use serde::{Serialize, Deserialize};

static COMMON_IDIOM_SET: OnceLock<HashSet<&'static str>> = OnceLock::new();

fn common_idioms() -> &'static HashSet<&'static str> {
    COMMON_IDIOM_SET.get_or_init(|| {
        HashSet::from([
            "一心一意",
            "全力以赴",
            "匠心独运",
            "万无一失",
            "一帆风顺",
            "百折不挠",
            "精益求精",
            "持之以恒",
            "开拓创新",
            "全神贯注",
            "言行一致",
            "一视同仁",
            "步步为营",
            "不言而喻",
            "博大精深",
            "不可或缺",
            "大公无私",
            "大吉大利",
            "得心应手",
            "独一无二",
            "防患未然",
            "凤凰于飞",
            "高瞻远瞩",
            "各抒己见",
            "恭喜发财",
            "果断决策",
            "海纳百川",
            "和风细雨",
            "和衷共济",
            "厚积薄发",
            "欢聚一堂",
            "坚如磐石",
            "举一反三",
            "精诚合作",
            "锦上添花",
            "尽善尽美",
            "井然有序",
            "刻不容缓",
            "来日方长",
            "力挽狂澜",
            "厉兵秣马",
            "立竿见影",
            "良师益友",
            "琳琅满目",
            "美轮美奂",
            "名列前茅",
            "铭记于心",
            "目不暇接",
            "逆风翻盘",
            "齐心协力",
            "潜移默化",
            "前所未有",
            "青出于蓝",
            "取长补短",
            "日新月异",
            "如虎添翼",
            "如期而至",
            "如鱼得水",
            "身临其境",
            "生机勃勃",
            "势在必行",
            "十全十美",
            "守望相助",
            "水滴石穿",
            "顺势而为",
            "所向披靡",
            "泰然自若",
            "天道酬勤",
            "同舟共济",
            "万众一心",
            "稳扎稳打",
            "无缝衔接",
            "无微不至",
            "细致入微",
            "先声夺人",
            "循序渐进",
            "扬帆起航",
            "一鼓作气",
            "一马当先",
            "一鸣惊人",
            "有条不紊",
            "有备无患",
            "与时俱进",
            "再接再厉",
            "蒸蒸日上",
            "众志成城",
            "卓有成效",
            "自强不息",
            "左右逢源",
            "一目了然",
            "别开生面",
            "沉着冷静",
            "出类拔萃",
            "大展宏图",
            "独具匠心",
            "风调雨顺",
            "功德无量",
            "厚德载物",
            "津津有味",
            "敬畏自然",
            "开门见山",
            "苦尽甘来",
            "理所当然",
            "两全其美",
            "满载而归",
            "妙手回春",
            "平步青云",
            "奇思妙想",
            "前程似锦",
            "热火朝天",
            "如火如荼",
            "如愿以偿",
            "深入人心",
            "十拿九稳",
            "事半功倍",
            "守口如瓶",
            "水乳交融",
            "推陈出新",
            "完美无瑕",
            "无与伦比",
            "喜闻乐见",
            "心照不宣",
            "兴高采烈",
            "意气风发",
            "迎刃而解",
            "与众不同",
            "展翅高飞",
            "振奋人心",
            "众望所归",
            "卓越不凡",
            "百花齐放",
            "百发百中",
            "春暖花开",
            "薪火相传",
            "精耕细作",
            "继往开来",
            "精雕细琢",
            "光明磊落",
            "奋发图强",
            "高歌猛进",
            "锦绣前程",
            "厚积成势",
            "稳中求进",
            "步步高升",
            "滴水穿石",
            "风生水起",
            "和睦相处",
            "和气生财",
            "浩气长存",
            "集思广益",
            "匡扶正义",
            "厚德载福",
            "激浊扬清",
            "破浪前行",
            "砥砺前行",
            "披荆斩棘",
            "日久弥新",
            "日进斗金",
            "守正创新",
            "四海升平",
            "挺身而出",
            "同心同德",
            "稳如泰山",
            "无往不利",
            "欣欣向荣",
            "雄姿英发",
            "学以致用",
            "勇往直前",
            "跃跃欲试",
            "自成一体",
            "众星捧月",
            "乘风破浪",
            "顶天立地",
            "福泽绵长",
            "高屋建瓴",
            "厚积薄发",
            "积极进取",
            "继往开来",
            "开疆拓土",
            "勠力同心",
            "披星戴月",
            "潜心钻研",
            "锐意进取",
            "同心协力",
            "无坚不摧",
            "心怀天下",
            "勇挑重担",
            "蒸蒸日上",
            "志在四方",
            "中流砥柱",
            "忠诚担当",
            "振兴中华",
            "百尺竿头",
            "乘势而上",
            "矢志不渝",
            "踔厉奋发",
            "继往开来",
            "昂首阔步",
            "奋发有为",
            "踔厉奋发",
            "砥砺奋进",
            "踔厉奋发",
            "团结奋进",
            "奋力拼搏",
            "振兴发展",
            "开创新局",
            "勇毅前行",
            "戮力同心",
            "阔步前行",
            "团结一心",
            "携手共进",
            "奋楫笃行",
            "勇敢无畏",
            "昂扬向上",
            "披荆斩棘",
            "破釜沉舟",
            "稳健发展",
            "群策群力",
            "一以贯之",
            "久久为功",
            "担当作为",
            "勇毅笃行",
            "踔厉奋发",
            "笃行不怠"
        ])
    })
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
        for (token, id) in &special_tokens {
            encode.insert(token.clone(), *id);
            decode.insert(*id, token.clone());
        }

        // Add remaining words starting from the next available ID
        let mut next_id = special_tokens.values().max().unwrap_or(&0) + 1;
        for word in words {
            let word_str = word.to_string();
            if !encode.contains_key(&word_str) {
                encode.insert(word_str.clone(), next_id);
                decode.insert(next_id, word_str.clone());
                next_id += 1;
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
        // Add default special tokens
        vocab_set.insert("<|pad|>".to_string());
        vocab_set.insert("<|unk|>".to_string());
        vocab_set.insert("<|bos|>".to_string());
        vocab_set.insert("</s>".to_string());  // End of sequence
        vocab_set.insert("<|sep|>".to_string());
        vocab_set.insert("<|cls|>".to_string());
        vocab_set.insert("<|mask|>".to_string());

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
        let mut chars = idiom.chars();
        if chars.clone().count() != 4 {
            return false;
        }
        if !chars.all(|c| c.is_chinese()) {
            return false;
        }
        common_idioms().contains(idiom)
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
