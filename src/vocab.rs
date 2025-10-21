//! # 词汇表管理模块（Vocabulary Management）
//!
//! 该模块负责管理语言模型的词汇表，是连接文本和神经网络的桥梁。
//!
//! ## 核心功能
//!
//! 1. **中文分词**：使用 jieba-rs 进行智能中文分词
//! 2. **词汇映射**：token 到 ID 的双向映射（编码/解码）
//! 3. **特殊词元管理**：处理 `<|pad|>`, `<|unk|>`, `</s>` 等特殊词元
//! 4. **成语识别**：检测并处理四字成语（如"一帆风顺"）
//! 5. **序列编码**：将文本转换为模型可处理的 token ID 序列
//!
//! ## 中文处理策略
//!
//! ### 为什么中文需要特殊处理？
//!
//! - **无空格分隔**：英文用空格分词，中文需要智能分词
//!   ```text
//!   英文: "I love programming"  → ["I", "love", "programming"]
//!   中文: "我爱编程"           → 需要分词器 → ["我", "爱", "编程"]
//!   ```
//!
//! - **多字词组**：中文的意义单元不是单个字，而是词组
//!   ```text
//!   错误分割: "人工智能" → ["人", "工", "智", "能"] (失去语义)
//!   正确分割: "人工智能" → ["人工智能"] (保留语义)
//!   ```
//!
//! - **成语识别**：四字成语是独立的语义单元
//!   ```text
//!   "一帆风顺" → 应作为一个词元，而非 ["一", "帆", "风", "顺"]
//!   ```
//!
//! ### Jieba 分词器
//!
//! 使用全局单例模式的 jieba-rs 分词器：
//! - **延迟初始化**：首次使用时才加载词典（减少启动时间）
//! - **全局共享**：避免重复初始化（节省内存）
//! - **线程安全**：使用 `OnceLock` 保证只初始化一次
//!
//! ## 词汇表结构
//!
//! ```text
//! Vocab {
//!     encode: HashMap<String, usize>    // 词 → ID 映射
//!     decode: HashMap<usize, String>    // ID → 词 映射
//!     words: Vec<String>                // 所有词的列表
//!     special_tokens: HashMap           // 特殊词元及其固定 ID
//! }
//! ```
//!
//! ## 特殊词元
//!
//! | 词元 | ID | 用途 |
//! |------|----|----|
//! | `<|pad|>` | 0 | 填充：补齐序列长度 |
//! | `<|unk|>` | 1 | 未知词：不在词汇表中的词 |
//! | `<|bos|>` | 2 | 开始标记：序列起始 |
//! | `</s>` | 3 | 结束标记：序列结束 |
//! | `<|sep|>` | 4 | 分隔符：分隔多个句子 |
//! | `<|cls|>` | 5 | 分类标记：用于分类任务 |
//! | `<|mask|>` | 6 | 掩码标记：用于 MLM 任务 |
//!
//! ## 编码流程示例
//!
//! ```text
//! 输入文本: "我爱人工智能"
//!
//! 步骤 1 - 检测语言:
//!   包含中文字符 (0x4E00-0x9FFF) → 使用 Jieba
//!
//! 步骤 2 - 分词:
//!   jieba.cut("我爱人工智能") → ["我", "爱", "人工智能"]
//!
//! 步骤 3 - 查表映射:
//!   "我"      → ID 102
//!   "爱"      → ID 358
//!   "人工智能" → ID 1524
//!
//! 步骤 4 - 输出:
//!   [102, 358, 1524]
//! ```
//!
//! ## 解码流程示例
//!
//! ```text
//! 输入 IDs: [102, 358, 1524]
//!
//! 步骤 1 - 反向查表:
//!   102  → "我"
//!   358  → "爱"
//!   1524 → "人工智能"
//!
//! 步骤 2 - 拼接:
//!   ["我", "爱", "人工智能"] → "我 爱 人工智能"
//!
//! 注意: 解码后中文词之间有空格，需要后处理移除
//! ```

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;
use std::sync::OnceLock;

use bincode::{Decode, Encode};
use jieba_rs::Jieba;
use regex::Regex;
use serde::{Deserialize, Serialize};

use std::fs;

/// **全局成语集合**
///
/// 使用 `OnceLock` 确保线程安全的延迟初始化：
/// - 首次调用时从 `data/chinese_idioms.json` 加载
/// - 后续调用直接返回已加载的实例
/// - 内存中只有一份拷贝，所有线程共享
static COMMON_IDIOM_SET: OnceLock<HashSet<String>> = OnceLock::new();

/// **全局 Jieba 分词器实例**
///
/// Jieba 初始化较慢（需要加载词典），使用全局单例避免重复初始化：
/// - **性能优势**：初始化一次 vs 每次分词都初始化
/// - **内存优势**：共享词典数据结构
/// - **线程安全**：`OnceLock` 保证只初始化一次
static JIEBA_INSTANCE: OnceLock<Jieba> = OnceLock::new();

/// **获取全局成语集合**
///
/// 延迟加载中文成语列表，用于成语识别和词汇构建。
///
/// # 返回值
/// 返回全局共享的成语 HashSet，包含从 JSON 文件加载的所有成语
///
/// # Panic
/// 如果无法加载 `data/chinese_idioms.json`，程序会 panic
fn common_idioms() -> &'static HashSet<String> {
    COMMON_IDIOM_SET.get_or_init(|| match load_common_idioms_from_file() {
        Ok(set) => set,
        Err(e) => {
            log::error!(
                "Failed to load chinese idioms from data/chinese_idioms.json: {}",
                e
            );
            HashSet::new()
        }
    })
}

/// **获取全局共享的 Jieba 分词器实例**
///
/// 延迟初始化 Jieba 分词器，避免重复加载词典。
///
/// # 为什么使用全局实例？
///
/// Jieba 初始化开销大（需要加载大型词典文件）：
/// - 每次创建新实例：~100ms 初始化时间
/// - 使用全局单例：只初始化一次，后续调用 ~0ms
///
/// # 线程安全
/// `OnceLock` 保证即使在多线程环境下也只初始化一次
fn jieba_instance() -> &'static Jieba {
    JIEBA_INSTANCE.get_or_init(|| {
        println!("⏳ 初始化全局 Jieba 分词器实例（仅一次）...");
        let jieba = Jieba::new();
        println!("✓ Jieba 分词器初始化完成！");
        jieba
    })
}

/// **从文件加载中文成语列表**
///
/// 从 `data/chinese_idioms.json` 读取成语数组并转换为 HashSet。
///
/// # 文件格式
/// ```json
/// [
///   "一帆风顺",
///   "水到渠成",
///   "画龙点睛",
///   ...
/// ]
/// ```
///
/// # 返回值
/// - `Ok(HashSet<String>)`: 成功加载的成语集合
/// - `Err`: 文件读取或 JSON 解析错误
fn load_common_idioms_from_file() -> Result<HashSet<String>, Box<dyn std::error::Error>> {
    let idioms_file_path = "data/chinese_idioms.json";
    let idioms_json = fs::read_to_string(idioms_file_path)?;
    let idioms: Vec<String> = serde_json::from_str(&idioms_json)?;
    Ok(HashSet::from_iter(idioms))
}

/// **词汇表结构体**
///
/// 存储词汇表的核心数据结构，提供双向映射（词↔ID）。
///
/// # 字段说明
///
/// - `encode`: 词 → ID 映射，用于编码（文本→数字）
/// - `decode`: ID → 词 映射，用于解码（数字→文本）
/// - `words`: 所有词的列表（包括特殊词元）
/// - `special_tokens`: 特殊词元及其固定 ID
///
/// # 序列化支持
///
/// 支持多种序列化格式：
/// - **JSON**: `serde_json` 用于人类可读的存储
/// - **Bincode**: `bincode` 用于高效的二进制存储
///
/// # 使用示例
///
/// ```rust
/// let vocab = Vocab::build_from_texts(&training_texts);
/// let token_ids = vocab.encode_sequence("我爱人工智能");
/// let text = vocab.decode_sequence(&token_ids);
/// ```
#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Vocab {
    /// **编码映射**: 词 → ID
    ///
    /// 查找复杂度: O(1)
    /// 示例: {"我": 102, "爱": 358, "人工智能": 1524}
    pub encode: HashMap<String, usize>,

    /// **解码映射**: ID → 词
    ///
    /// 查找复杂度: O(1)
    /// 示例: {102: "我", 358: "爱", 1524: "人工智能"}
    pub decode: HashMap<usize, String>,

    /// **词列表**: 所有词的向量
    ///
    /// 用于遍历整个词汇表或统计词汇量
    pub words: Vec<String>,

    /// **特殊词元映射**: 特殊词元名称 → 固定 ID
    ///
    /// 示例: {"<|pad|>": 0, "<|unk|>": 1, "</s>": 3}
    pub special_tokens: HashMap<String, usize>,
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new_with_special_tokens(Self::default_words(), Self::default_special_tokens())
    }
}

impl Vocab {
    /// **创建新的词汇表**
    ///
    /// 从词列表创建词汇表，使用默认的特殊词元配置。
    ///
    /// # 参数
    /// - `words`: 词列表（不包括特殊词元）
    ///
    /// # 返回值
    /// 新创建的词汇表实例
    ///
    /// # 示例
    /// ```rust
    /// let vocab = Vocab::new(vec!["你好", "世界", "人工智能"]);
    /// ```
    pub fn new(words: Vec<&str>) -> Self {
        Self::new_with_special_tokens(words, Self::default_special_tokens())
    }

    /// **创建带自定义特殊词元的词汇表**
    ///
    /// 这是词汇表构建的核心方法，处理词汇映射的创建逻辑。
    ///
    /// # 构建流程
    ///
    /// 1. **添加特殊词元**：按预定义 ID 添加（如 `<|pad|>` → 0）
    /// 2. **添加常规词汇**：从 ID=7 开始递增分配
    /// 3. **去重处理**：跳过已存在的词，避免 ID 冲突
    /// 4. **统计信息**：输出中文词、英文词、重复词的数量
    ///
    /// # 参数
    /// - `words`: 常规词汇列表
    /// - `special_tokens`: 特殊词元及其固定 ID
    ///
    /// # 返回值
    /// 完整的词汇表实例
    ///
    /// # ID 分配规则
    ///
    /// ```text
    /// ID 0-6:  特殊词元（固定分配）
    /// ID 7+:   常规词汇（动态分配，按添加顺序）
    /// ```
    pub fn new_with_special_tokens(
        words: Vec<&str>,
        special_tokens: HashMap<String, usize>,
    ) -> Self {
        let mut encode = HashMap::new();
        let mut decode = HashMap::new();

        // Add special tokens first with their predefined IDs
        println!("\n=== 初始化词汇表：添加特殊词元 ===");
        let mut special_tokens_sorted: Vec<_> = special_tokens.iter().collect();
        special_tokens_sorted.sort_by_key(|(_, id)| *id);

        for (token, id) in &special_tokens_sorted {
            encode.insert((*token).clone(), **id);
            decode.insert(**id, (*token).clone());
            println!("  ✓ 特殊词元: '{}' -> ID {}", token, id);
        }
        println!("特殊词元添加完成，共 {} 个\n", special_tokens.len());

        // Add remaining words starting from the next available ID
        println!("=== 添加常规词汇 ===");
        let mut next_id = special_tokens.values().max().unwrap_or(&0) + 1;
        let mut added_count = 0;
        let mut skipped_count = 0;
        let mut chinese_count = 0;
        let mut english_count = 0;

        // 统计词元类型
        for word in words.iter() {
            let word_str = word.to_string();
            if !encode.contains_key(&word_str) {
                encode.insert(word_str.clone(), next_id);
                decode.insert(next_id, word_str.clone());

                // 判断词元类型
                let is_chinese = word_str
                    .chars()
                    .any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);
                if is_chinese {
                    chinese_count += 1;
                    println!("  [中文] '{}' -> ID {}", word_str, next_id);
                } else {
                    english_count += 1;
                    println!("  [其他] '{}' -> ID {}", word_str, next_id);
                }

                added_count += 1;
                next_id += 1;
            } else {
                skipped_count += 1;
                println!(
                    "  [跳过] '{}' (已存在，ID: {})",
                    word_str, encode[&word_str]
                );
            }
        }

        println!("\n=== 词汇表构建完成 ===");
        println!("  • 新增词元: {} 个", added_count);
        println!("  • 跳过重复: {} 个", skipped_count);
        println!("  • 中文词元: {} 个", chinese_count);
        println!("  • 其他词元: {} 个", english_count);
        println!("  • 总词汇量: {} 个", encode.len());
        println!("========================\n");

        let all_words: Vec<String> = decode.values().cloned().collect();

        Vocab {
            encode,
            decode,
            words: all_words,
            special_tokens,
        }
    }

    /// **编码单个词**
    ///
    /// 将词转换为 token ID。
    ///
    /// # 参数
    /// - `word`: 要编码的词
    ///
    /// # 返回值
    /// - `Some(usize)`: 词对应的 ID
    /// - `None`: 词不在词汇表中
    ///
    /// # 示例
    /// ```rust
    /// if let Some(id) = vocab.encode("你好") {
    ///     println!("'你好' 的 ID 是: {}", id);
    /// }
    /// ```
    pub fn encode(&self, word: &str) -> Option<usize> {
        self.encode.get(word).copied()
    }

    /// **编码单个词（带未知词回退）**
    ///
    /// 将词转换为 token ID，如果词不在词汇表中则返回 `<|unk|>` 的 ID。
    ///
    /// # 参数
    /// - `word`: 要编码的词
    ///
    /// # 返回值
    /// 词对应的 ID，如果不存在则返回 `<|unk|>` 的 ID (通常是 1)
    ///
    /// # 为什么需要未知词处理？
    ///
    /// 训练后的词汇表是固定的，但推理时可能遇到新词：
    /// - 生僻字、专有名词
    /// - 新出现的网络用语
    /// - 拼写错误的词
    ///
    /// 使用 `<|unk|>` 让模型能够处理这些情况，而不是崩溃。
    ///
    /// # 示例
    /// ```rust
    /// let id = vocab.encode_with_unk("火星文词汇"); // 返回 1 (<|unk|>)
    /// ```
    pub fn encode_with_unk(&self, word: &str) -> usize {
        match self.encode(word) {
            Some(id) => id,
            None => *self.special_tokens.get("<|unk|>").unwrap_or(&0),
        }
    }

    /// **解码单个 token ID**
    ///
    /// 将 token ID 转换回对应的词。
    ///
    /// # 参数
    /// - `token_id`: 要解码的 token ID
    ///
    /// # 返回值
    /// - `Some(&String)`: ID 对应的词
    /// - `None`: ID 不在词汇表中（不应该发生）
    #[allow(dead_code)]
    pub fn decode(&self, token_id: usize) -> Option<&String> {
        self.decode.get(&token_id)
    }

    /// **编码文本序列**
    ///
    /// 将整段文本转换为 token ID 序列，这是模型输入的标准格式。
    ///
    /// # 算法流程
    ///
    /// ```text
    /// 1. 检测语言（是否包含中文字符 0x4E00-0x9FFF）
    ///
    /// 2a. 如果是中文：
    ///     - 使用 Jieba 分词: "我爱编程" → ["我", "爱", "编程"]
    ///     - 查表映射: ["我", "爱", "编程"] → [102, 358, 456]
    ///
    /// 2b. 如果是英文：
    ///     - 按空格分词: "I love coding" → ["I", "love", "coding"]
    ///     - 查表映射: ["I", "love", "coding"] → [78, 234, 567]
    ///
    /// 3. 返回 token ID 序列: [102, 358, 456]
    /// ```
    ///
    /// # 参数
    /// - `text`: 要编码的文本
    ///
    /// # 返回值
    /// token ID 序列
    ///
    /// # 示例
    /// ```rust
    /// let text = "深度学习很有趣";
    /// let token_ids = vocab.encode_sequence(text);
    /// // token_ids: [1234, 5678, 9012, 3456]
    /// ```
    pub fn encode_sequence(&self, text: &str) -> Vec<usize> {
        let mut tokens = Vec::new();

        // Check if the text contains Chinese characters
        let has_chinese = text
            .chars()
            .any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);

        if has_chinese {
            let jieba = jieba_instance(); // 使用全局实例
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

    /// **解码 token ID 序列**
    ///
    /// 将 token ID 序列转换回文本，这是模型输出的标准格式。
    ///
    /// # 解码流程
    ///
    /// ```text
    /// 输入: [102, 358, 1524]
    ///   ↓ 查表
    /// ["我", "爱", "人工智能"]
    ///   ↓ 用空格连接
    /// "我 爱 人工智能"
    /// ```
    ///
    /// # 注意事项
    ///
    /// - **空格问题**：解码后的中文词之间会有空格
    /// - **后处理**：需要在 `llm.rs` 中移除中文词之间的空格
    /// - **特殊词元**：`<|pad|>`, `<|unk|>` 等也会被解码出来
    ///
    /// # 参数
    /// - `token_ids`: token ID 序列
    ///
    /// # 返回值
    /// 解码后的文本（词之间用空格分隔）
    ///
    /// # 示例
    /// ```rust
    /// let token_ids = vec![102, 358, 1524];
    /// let text = vocab.decode_sequence(&token_ids);
    /// // text: "我 爱 人工智能"
    /// ```
    pub fn decode_sequence(&self, token_ids: &[usize]) -> String {
        let mut result = Vec::new();
        for &token_id in token_ids {
            if let Some(word) = self.decode.get(&token_id) {
                result.push(word.clone());
            }
        }
        result.join(" ")
    }

    /// **获取词汇表大小**
    ///
    /// 返回词汇表中的词汇总数（包括特殊词元）。
    ///
    /// # 返回值
    /// 词汇量（通常在 5,000 到 30,000 之间）
    pub fn len(&self) -> usize {
        self.encode.len()
    }

    /// **检查词汇表是否为空**
    pub fn is_empty(&self) -> bool {
        self.encode.is_empty()
    }

    /// **获取未知词的 token ID**
    ///
    /// 返回 `<|unk|>` 的 ID，通常是 1。
    pub fn unk_token_id(&self) -> usize {
        *self.special_tokens.get("<|unk|>").unwrap_or(&0)
    }

    /// **获取填充词的 token ID**
    ///
    /// 返回 `<|pad|>` 的 ID，通常是 0。
    pub fn pad_token_id(&self) -> usize {
        *self.special_tokens.get("<|pad|>").unwrap_or(&0)
    }

    /// **获取序列结束词的 token ID**
    ///
    /// 返回 `</s>` 的 ID，通常是 3。
    pub fn eos_token_id(&self) -> usize {
        *self.special_tokens.get("</s>").unwrap_or(&0)
    }

    /// **获取序列开始词的 token ID**
    ///
    /// 返回 `<|bos|>` 的 ID，通常是 2。
    pub fn bos_token_id(&self) -> usize {
        *self.special_tokens.get("<|bos|>").unwrap_or(&0)
    }

    /// **默认词列表**
    ///
    /// 返回一个小型英文词汇列表，主要用于测试。
    ///
    /// 实际训练中，词汇表会从训练数据动态构建。
    pub fn default_words() -> Vec<&'static str> {
        vec![
            "hello", "world", "this", "is", "rust", "the", "a", "an", "and", "or", "but", "in",
            "on", "at", "to", "for", "of", "with", "by", "from", "up", "about", "into", "through",
            "during", "before", "after", "above", "below", "between", "among", "as", "if", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
            "can", "will", "just", "don", "should", "now",
        ]
    }

    /// **默认特殊词元配置**
    ///
    /// 返回标准的特殊词元映射，这些 ID 是固定的。
    ///
    /// # 特殊词元列表
    ///
    /// | 词元 | ID | 用途 |
    /// |------|----|----|
    /// | `<|pad|>` | 0 | 填充短序列 |
    /// | `<|unk|>` | 1 | 未知词占位符 |
    /// | `<|bos|>` | 2 | 开始标记 |
    /// | `</s>` | 3 | 结束标记 |
    /// | `<|sep|>` | 4 | 分隔符 |
    /// | `<|cls|>` | 5 | 分类标记 |
    /// | `<|mask|>` | 6 | 掩码标记 |
    pub fn default_special_tokens() -> HashMap<String, usize> {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|pad|>".to_string(), 0);
        special_tokens.insert("<|unk|>".to_string(), 1);
        special_tokens.insert("<|bos|>".to_string(), 2);
        special_tokens.insert("</s>".to_string(), 3); // End of sequence
        special_tokens.insert("<|sep|>".to_string(), 4); // Separator
        special_tokens.insert("<|cls|>".to_string(), 5); // Classification
        special_tokens.insert("<|mask|>".to_string(), 6); // Masked token
        special_tokens
    }

    /// **从文本数据构建词汇表**
    ///
    /// 这是词汇表构建的主入口函数，处理训练数据并提取所有唯一词元。
    ///
    /// # 处理流程
    ///
    /// ```text
    /// 1. 初始化词汇集合，添加特殊词元
    ///
    /// 2. 对每个训练文本：
    ///    a. 检测语言（中文 vs 英文）
    ///    b. 中文：使用 Jieba 分词
    ///    c. 英文：按空格分词
    ///    d. 提取成语和有意义的短语
    ///    e. 添加到词汇集合
    ///
    /// 3. 统计并输出：
    ///    - 中文词元数量
    ///    - 英文词元数量
    ///    - 最终词汇表大小
    ///
    /// 4. 创建词汇表实例
    /// ```
    ///
    /// # 参数
    /// - `texts`: 所有训练文本（包括预训练和对话数据）
    /// - `vocab_set`: 词汇集合（会被就地修改）
    ///
    /// # 性能考虑
    ///
    /// - 使用 `HashSet` 自动去重，O(1) 插入和查找
    /// - Jieba 分词较慢，但只在构建阶段运行一次
    /// - 成语识别使用正则表达式，性能良好
    ///
    /// # 示例输出
    ///
    /// ```text
    /// 🔧 开始处理文本数据以构建词汇表...
    ///   📊 待处理文本数量: 1000
    ///   ✓ 已添加 7 个特殊词元
    ///
    /// 📝 开始分词处理...
    ///   📄 处理文本 [1/1000]
    ///      内容预览: 深度学习是人工智能的重要分支...
    ///      类型: 中文文本
    ///      ⏳ 开始 Jieba 分词...
    ///      ✓ 分词完成，提取了 15 个词元
    ///
    /// ✅ 文本处理完成！
    /// 📊 分词处理统计:
    ///   • 处理文本总数: 1000 个
    ///   • 中文文本: 950 个
    ///   • 其他文本: 50 个
    ///   • 最终词汇集大小: 12500 个唯一词元
    /// ```
    pub fn process_text_for_vocab(texts: &[String], vocab_set: &mut HashSet<String>) {
        use std::io::Write;

        println!("\n🔧 开始处理文本数据以构建词汇表...");
        println!("  📊 待处理文本数量: {}", texts.len());

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
        println!("  ✓ 已添加 7 个特殊词元");

        // 使用全局 Jieba 实例（如果未初始化会自动初始化）
        println!("\n📝 开始分词处理...");
        let jieba = jieba_instance();

        // Process all training examples for vocabulary
        let total_texts = texts.len();
        let mut chinese_texts = 0;
        let mut english_texts = 0;

        for (idx, text) in texts.iter().enumerate() {
            // 显示当前处理的文本进度
            println!("\n  📄 处理文本 [{}/{}]", idx + 1, total_texts);

            // 安全地截取文本预览（处理 UTF-8 字符边界）
            let preview = if text.len() > 50 {
                // 使用字符迭代器确保不会在字符中间切割
                text.chars().take(50).collect::<String>()
            } else {
                text.clone()
            };
            println!("     内容预览: {}...", preview);
            if let Err(e) = std::io::stdout().flush() {
                log::warn!("刷新标准输出失败: {}", e);
            }

            // Check if the text contains Chinese characters
            let has_chinese = text
                .chars()
                .any(|c| (c as u32) >= 0x4E00 && (c as u32) <= 0x9FFF);

            if has_chinese {
                chinese_texts += 1;
                println!("     类型: 中文文本");

                // Use Jieba for Chinese text tokenization
                println!("     ⏳ 开始 Jieba 分词...");
                if let Err(e) = std::io::stdout().flush() {
                    log::warn!("刷新标准输出失败: {}", e);
                }

                let tokens = jieba.cut(text, false);
                let token_count = tokens.len();

                println!("     ✓ 分词完成，提取了 {} 个词元", token_count);

                for token in tokens {
                    if !token.trim().is_empty() {
                        let token_trimmed = token.trim().to_string();
                        vocab_log.push_str(&format!("Token: {}\n", token_trimmed));
                        let is_new = vocab_set.insert(token_trimmed.clone());
                        if is_new {
                            println!("       + 新词元: '{}'", token_trimmed);
                        }
                    }
                }

                // Process common Chinese idioms and phrases that might be missed by Jieba
                println!("     🔍 提取成语和短语...");
                Self::extract_chinese_phrases(text, vocab_set);
            } else {
                english_texts += 1;
                println!("     类型: 英文/其他文本");

                // Use the original method for non-Chinese text
                for word in text.split_whitespace() {
                    // Handle punctuation by splitting it from words
                    let mut current = String::new();
                    for c in word.chars() {
                        if c.is_ascii_punctuation() {
                            if !current.is_empty() {
                                vocab_log.push_str(&format!("Word: {}\n", current));
                                let is_new = vocab_set.insert(current.clone());
                                if is_new {
                                    println!("       + 新词元: '{}'", current);
                                }
                                current.clear();
                            }
                            vocab_log.push_str(&format!("Punctuation: {}\n", c));
                            let is_new = vocab_set.insert(c.to_string());
                            if is_new {
                                println!("       + 新标点: '{}'", c);
                            }
                        } else {
                            current.push(c);
                        }
                    }
                    if !current.is_empty() {
                        vocab_log.push_str(&format!("Word: {}\n", current));
                        let is_new = vocab_set.insert(current.clone());
                        if is_new {
                            println!("       + 新词元: '{}'", current);
                        }
                    }
                }
            }

            println!("     📊 当前词汇表大小: {} 个唯一词元", vocab_set.len());
        }

        // 显示最终统计
        println!("\n✅ 文本处理完成！");
        println!("\n📊 分词处理统计:");
        println!("  • 处理文本总数: {} 个", total_texts);
        println!("  • 中文文本: {} 个", chinese_texts);
        println!("  • 其他文本: {} 个", english_texts);
        println!("  • 最终词汇集大小: {} 个唯一词元", vocab_set.len());

        let _ = writeln!(idiom_writer, "{}", vocab_log);
    }

    /// **提取中文短语和成语**
    ///
    /// 使用正则表达式识别中文成语和有意义的短语。
    ///
    /// # 识别策略
    ///
    /// 1. **四字成语检测**：
    ///    - 正则: `[\u4e00-\u9fff]{4}` 匹配4个连续中文字符
    ///    - 验证: 在成语字典中查找（`data/chinese_idioms.json`）
    ///    - 示例: "一帆风顺", "画龙点睛"
    ///
    /// 2. **有意义短语检测**：
    ///    - 正则: `[\u4e00-\u9fff]{2,6}` 匹配2-6个中文字符
    ///    - 验证: 使用 Jieba 检查是否为独立词组
    ///    - 示例: "人工智能", "深度学习"
    ///
    /// # 为什么需要这个？
    ///
    /// Jieba 有时会把成语拆分成单字，导致语义丢失：
    /// ```text
    /// 错误: "一帆风顺" → ["一", "帆", "风", "顺"]
    /// 正确: "一帆风顺" → ["一帆风顺"]
    /// ```
    ///
    /// # 参数
    /// - `text`: 待处理的文本
    /// - `vocab_set`: 词汇集合（会添加识别出的短语）
    fn extract_chinese_phrases(text: &str, vocab_set: &mut HashSet<String>) {
        // Common Chinese idioms (四字成语) - these are often not segmented properly by Jieba
        let idiom_regex = match Regex::new(r"[\u4e00-\u9fff]{4}") {
            Ok(re) => re,
            Err(e) => {
                log::warn!("成语正则编译失败: {}，跳过成语提取", e);
                return;
            }
        };
        for mat in idiom_regex.find_iter(text) {
            let idiom = mat.as_str();
            if Self::is_common_chinese_idiom(idiom) {
                vocab_set.insert(idiom.to_string());
            }
        }

        // Common multi-character phrases that might be relevant
        let phrase_regex = match Regex::new(r"[\u4e00-\u9fff]{2,6}") {
            Ok(re) => re,
            Err(e) => {
                log::warn!("短语正则编译失败: {}，跳过短语提取", e);
                return;
            }
        };
        for mat in phrase_regex.find_iter(text) {
            let phrase = mat.as_str();
            if Self::is_meaningful_phrase(phrase) {
                vocab_set.insert(phrase.to_string());
            }
        }
    }

    /// **检查是否为常见中文成语**
    ///
    /// 验证4字符串是否在成语字典中。
    ///
    /// # 验证规则
    ///
    /// 1. 必须是4个字符
    /// 2. 所有字符必须是中文（0x4E00-0x9FFF）
    /// 3. 在加载的成语字典中存在
    ///
    /// # 参数
    /// - `idiom`: 待检查的字符串
    ///
    /// # 返回值
    /// - `true`: 是常见成语
    /// - `false`: 不是成语或不符合规则
    fn is_common_chinese_idiom(idiom: &str) -> bool {
        let mut chars = idiom.chars();
        if chars.clone().count() != 4 {
            return false;
        }
        if !chars.all(|c| c.is_chinese()) {
            return false;
        }
        common_idioms().contains(idiom) // This works because HashSet<String> implements contains for &str
    }

    /// **检查是否为有意义的中文短语**
    ///
    /// 使用 Jieba 分词结果判断短语是否为独立的语义单元。
    ///
    /// # 判断规则
    ///
    /// 1. 长度在 2-8 个字符之间
    /// 2. 所有字符都是中文
    /// 3. Jieba 将其识别为一个或两个词组
    ///
    /// # 示例
    ///
    /// ```text
    /// "人工智能" → Jieba: ["人工智能"] → 1个词 → true
    /// "深度学习" → Jieba: ["深度", "学习"] → 2个词，总长度匹配 → true
    /// "的是在" → Jieba: ["的", "是", "在"] → 3个词 → false
    /// ```
    ///
    /// # 参数
    /// - `phrase`: 待检查的短语
    ///
    /// # 返回值
    /// - `true`: 有意义的短语
    /// - `false`: 不是有意义的短语
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
        let jieba = jieba_instance(); // 使用全局实例
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

    /// **从文本列表构建词汇表**
    ///
    /// 高级接口：处理所有文本数据并返回完整的词汇表实例。
    ///
    /// # 构建步骤
    ///
    /// 1. 初始化空的词汇集合
    /// 2. 调用 `process_text_for_vocab` 提取所有词元
    /// 3. 排序词汇（确保确定性顺序）
    /// 4. 创建词汇表实例
    ///
    /// # 参数
    /// - `texts`: 所有训练文本
    ///
    /// # 返回值
    /// 完整的词汇表实例，包含：
    /// - 特殊词元（ID 0-6）
    /// - 从数据中提取的所有唯一词元（ID 7+）
    ///
    /// # 使用示例
    ///
    /// ```rust
    /// let training_texts = load_training_data();
    /// let vocab = Vocab::build_from_texts(&training_texts);
    /// println!("词汇表大小: {}", vocab.len());
    /// ```
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

    /// **保存词汇表到文件**
    ///
    /// 将词汇表序列化为 JSON 格式并保存到文件。
    ///
    /// # 参数
    /// - `path`: 保存路径
    ///
    /// # 返回值
    /// - `Ok(())`: 保存成功
    /// - `Err`: 文件写入错误
    ///
    /// # 使用场景
    ///
    /// 训练完成后保存词汇表，推理时可以直接加载，避免重新构建。
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        Ok(())
    }

    /// **从文件加载词汇表**
    ///
    /// 从 JSON 文件反序列化词汇表。
    ///
    /// # 参数
    /// - `path`: 词汇表文件路径
    ///
    /// # 返回值
    /// - `Ok(Vocab)`: 加载成功的词汇表
    /// - `Err`: 文件读取或 JSON 解析错误
    ///
    /// # 使用场景
    ///
    /// 推理时加载已保存的词汇表，避免重新构建（节省时间）。
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let vocab: Vocab = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(vocab)
    }

    /// **动态添加词汇**
    ///
    /// 在词汇表创建后添加新词。
    ///
    /// # 参数
    /// - `word`: 要添加的词
    ///
    /// # 返回值
    /// 新词的 ID（如果已存在，返回现有 ID）
    ///
    /// # 注意
    ///
    /// 动态添加词会改变词汇表大小，可能导致：
    /// - 嵌入层需要扩展
    /// - 输出投影层需要扩展
    /// - 模型需要重新训练
    ///
    /// 因此，一般只在构建阶段使用，训练后不建议添加新词。
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

/// **辅助 trait：检查字符是否为中文**
///
/// 为 `char` 类型添加 `is_chinese()` 方法，简化中文字符判断。
///
/// # Unicode 范围
///
/// 中文字符的 Unicode 范围是 U+4E00 到 U+9FFF（CJK 统一表意文字）：
/// - 包含常用简体字和繁体字
/// - 不包括标点符号和特殊符号
///
/// # 使用示例
///
/// ```rust
/// let c = '中';
/// if c.is_chinese() {
///     println!("这是中文字符");
/// }
/// ```
trait IsChinese {
    fn is_chinese(&self) -> bool;
}

impl IsChinese for char {
    fn is_chinese(&self) -> bool {
        (*self as u32) >= 0x4E00 && (*self as u32) <= 0x9FFF
    }
}

/// **将词汇表转换为字符串**
///
/// 用于调试和日志输出，显示词汇表的所有词及其 ID。
///
/// # 格式
///
/// ```text
/// (0,<|pad|>),(1,<|unk|>),(2,<|bos|>),...
/// ```
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
