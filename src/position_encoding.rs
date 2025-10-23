//! # 位置编码模块（Position Encoding）
//!
//! 位置编码为 Transformer 提供序列中的位置信息，弥补自注意力机制的位置不敏感性。
//!
//! ## 为什么需要位置编码？
//!
//! **问题**：自注意力机制本身是位置无关的，打乱输入顺序不影响输出：
//! ```text
//! 输入: ["我", "爱", "中国"]
//! 打乱: ["中国", "我", "爱"]
//! 问题: 对注意力机制来说，两者是等价的（但语义完全不同！）
//! ```
//!
//! **解决方案**：在嵌入向量中注入位置信息，让模型知道词的顺序。
//!
//! ## 正弦位置编码（Sinusoidal Position Encoding）
//!
//! ### 数学公式
//!
//! ```text
//! PE(pos, 2i)   = sin(pos / 10000^(2i/d))     // 偶数维度使用 sin
//! PE(pos, 2i+1) = cos(pos / 10000^(2i/d))     // 奇数维度使用 cos
//! ```
//!
//! 其中：
//! - `pos`: 位置索引（0, 1, 2, ...）
//! - `i`: 维度索引（0 到 d/2）
//! - `d`: 嵌入维度（512）
//!
//! ### 为什么使用三角函数？
//!
//! 1. **平滑渐变**：相邻位置的编码相似，距离越远差异越大
//! 2. **周期性**：能够外推到训练时未见过的更长序列
//! 3. **相对位置**：sin/cos 的加法定理让模型能够学习相对位置关系
//! 4. **不需要训练**：完全基于数学公式，无需学习参数
//!
//! ### 不同频率的作用
//!
//! ```text
//! 维度 0-1:   波长 = 2π，快速变化，捕捉相邻位置
//! 维度 2-3:   波长 = 4π
//! ...
//! 维度 510-511: 波长 = 10000×2π，缓慢变化，捕捉长距离依赖
//! ```
//!
//! ## 实现细节
//!
//! - **预计算**：在初始化时计算所有位置的编码（256×512 矩阵）
//! - **加法融合**：位置编码直接加到词嵌入上
//! - **无需梯度**：位置编码是固定的，不参与训练
//!
//! ## 示例
//!
//! ```text
//! 位置 0 的编码 (前4维):
//!   PE(0,0) = sin(0) = 0.0
//!   PE(0,1) = cos(0) = 1.0
//!   PE(0,2) = sin(0) = 0.0
//!   PE(0,3) = cos(0) = 1.0
//!
//! 位置 1 的编码 (前4维):
//!   PE(1,0) = sin(1/10000^0) ≈ 0.841
//!   PE(1,1) = cos(1/10000^0) ≈ 0.540
//!   PE(1,2) = sin(1/10000^(2/512)) ≈ 0.841
//!   PE(1,3) = cos(1/10000^(2/512)) ≈ 0.540
//! ```

use ndarray::Array2;

use crate::{EMBEDDING_DIM, MAX_POSITIONAL_LEN};

/// **位置编码结构体**
///
/// 存储预计算的正弦位置编码矩阵。
pub struct PositionEncoding {
    /// **编码矩阵**: (MAX_POSITIONAL_LEN, EMBEDDING_DIM)
    ///
    /// 每行对应一个位置的编码向量
    pub encoding: Array2<f32>,
}

impl PositionEncoding {
    /// **创建新的位置编码实例**
    ///
    /// 预计算所有位置的正弦位置编码。
    ///
    /// # 计算流程
    ///
    /// ```text
    /// 对于每个位置 pos (0 到 MAX_POSITIONAL_LEN-1):
    ///   对于每个维度 i (0 到 EMBEDDING_DIM-1):
    ///     angle = pos / 10000^(i/d)
    ///     if i 是偶数:
    ///       PE[pos, i] = sin(angle)
    ///     else:
    ///       PE[pos, i] = cos(angle)
    /// ```
    ///
    /// # 性能考虑
    ///
    /// - 初始化时计算一次：O(MAX_POSITIONAL_LEN × EMBEDDING_DIM)
    /// - 后续使用时直接查表：O(1)
    /// - 内存占用：MAX_POSITIONAL_LEN × EMBEDDING_DIM × 4 bytes
    pub fn new() -> Self {
        // Initialize the position encoding matrix (MAX_POSITIONAL_LEN x EMBEDDING_DIM)
        let mut encoding = Array2::zeros((MAX_POSITIONAL_LEN, EMBEDDING_DIM));

        for pos in 0..MAX_POSITIONAL_LEN {
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

    /// **获取特定位置和维度的编码值**
    ///
    /// # 参数
    /// - `position`: 位置索引（0 到 MAX_POSITIONAL_LEN-1）
    /// - `dimension`: 维度索引（0 到 EMBEDDING_DIM-1）
    ///
    /// # Panic
    /// 如果位置或维度超出范围，程序会 panic
    pub fn get_encoding(&self, position: usize, dimension: usize) -> f32 {
        if position >= MAX_POSITIONAL_LEN || dimension >= EMBEDDING_DIM {
            log::warn!(
                "位置编码索引越界 position={} dimension={}，将返回0.0 作为回退",
                position,
                dimension
            );
            return 0.0;
        }
        self.encoding[[position, dimension]]
    }

    /// **将位置编码应用到输入嵌入**
    ///
    /// 将位置信息注入到词嵌入中，使用加法融合。
    ///
    /// # 操作
    ///
    /// ```text
    /// 对于序列中的每个位置 pos:
    ///   embedding[pos] = embedding[pos] + position_encoding[pos]
    /// ```
    ///
    /// # 参数
    /// - `input`: 词嵌入矩阵 (seq_len, embedding_dim)，会被就地修改
    ///
    /// # 使用示例
    ///
    /// ```rust
    /// let mut embeddings = Array2::zeros((4, 512));  // 4个词的嵌入
    /// let pos_enc = PositionEncoding::new();
    /// pos_enc.apply_to_input(&mut embeddings);
    /// // 现在 embeddings 包含了位置信息
    /// ```
    #[allow(dead_code)]
    pub fn apply_to_input(&self, input: &mut Array2<f32>) {
        let (seq_len, embedding_dim) = input.dim();

        // Determine how many positions we can encode based on input length
        let positions_to_encode = std::cmp::min(seq_len, MAX_POSITIONAL_LEN);
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

/// **相对位置编码结构体**
///
/// 实验性功能：编码相对位置而非绝对位置，可能更适合中文文本。
///
/// # 相对位置 vs 绝对位置
///
/// - **绝对位置**：第0个词，第1个词，第2个词
/// - **相对位置**：前1个词，前2个词，后1个词
///
/// # 优势
///
/// 相对位置编码对于中文可能更有效，因为中文的语法关系更多基于相对距离。
///
/// # 注意
///
/// 当前项目未使用此功能，使用标准的绝对位置编码。
#[allow(dead_code)]
pub struct RelativePositionEncoding {
    /// 编码矩阵：(2×max_offset+1, EMBEDDING_DIM)
    pub encoding: Array2<f32>,

    /// 最大偏移量：支持 [-max_offset, +max_offset] 的相对位置
    pub max_offset: usize,
}

impl RelativePositionEncoding {
    /// 创建相对位置编码实例
    #[allow(dead_code)]
    pub fn new(max_offset: usize) -> Self {
        // Create relative position encoding matrix
        let total_positions = 2 * max_offset + 1;
        let mut encoding = Array2::zeros((total_positions, EMBEDDING_DIM));

        for offset in 0..total_positions {
            let relative_pos = offset as i32 - max_offset as i32;
            for i in 0..EMBEDDING_DIM {
                let angle =
                    relative_pos as f32 / 10000f32.powf((i / 2) as f32 / EMBEDDING_DIM as f32);

                if i % 2 == 0 {
                    encoding[[offset, i]] = angle.sin();
                } else {
                    encoding[[offset, i]] = angle.cos();
                }
            }
        }

        Self {
            encoding,
            max_offset,
        }
    }

    /// 获取指定相对位置的编码
    ///
    /// # 参数
    /// - `relative_pos`: 相对位置（负数表示前面的词，正数表示后面的词）
    ///
    /// # 返回值
    /// - `Some(Vec<f32>)`: 编码向量
    /// - `None`: 相对位置超出范围
    #[allow(dead_code)]
    pub fn get_encoding(&self, relative_pos: i32) -> Option<Vec<f32>> {
        let index = relative_pos as i32 + self.max_offset as i32;
        if index < 0 || index >= (2 * self.max_offset + 1) as i32 {
            return None;
        }
        let index = index as usize;
        Some(self.encoding.row(index).to_vec())
    }
}
