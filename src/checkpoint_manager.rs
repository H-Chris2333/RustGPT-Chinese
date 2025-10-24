//! 检查点管理器模块
//!
//! 提供训练检查点的保存、加载和管理功能，支持多种保存策略：
//! - **Best**: 保存最佳模型（基于loss）
//! - **Last**: 保存最新模型
//! - **Periodic**: 周期性保存（每N个epoch）
//!
//! 检查点包含完整的训练状态：
//! - 模型参数（所有层的权重）
//! - 优化器状态（Adam的m、v动量和timestep）
//! - 训练元数据（epoch、loss、学习率等）
//! - 词汇表

use std::fs;
use std::path::{Path, PathBuf};

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::llm::LLM;
use crate::model_serialization::SerializableModel;

/// 检查点元数据
#[derive(Clone, Debug, Encode, Decode, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// 当前epoch
    pub epoch: usize,
    /// 平均loss
    pub loss: f32,
    /// 当前学习率
    pub learning_rate: f32,
    /// 保存时间戳
    pub timestamp: String,
    /// 训练阶段标识（如"pretraining", "instruction_tuning"）
    pub phase: String,
}

/// 完整的检查点数据
#[derive(Clone, Encode, Decode, Serialize, Deserialize)]
pub struct Checkpoint {
    /// 模型状态
    pub model: SerializableModel,
    /// 元数据
    pub metadata: CheckpointMetadata,
}

/// 检查点保存策略
#[derive(Clone, Debug)]
pub enum CheckpointStrategy {
    /// 仅保存最佳模型
    Best,
    /// 保存最新模型
    Last,
    /// 周期性保存（每N个epoch）
    Periodic(usize),
    /// 组合策略：最佳 + 最新
    BestAndLast,
    /// 组合策略：最佳 + 周期性
    BestAndPeriodic(usize),
}

/// 检查点管理器
pub struct CheckpointManager {
    /// 检查点保存目录
    checkpoint_dir: PathBuf,
    /// 保存策略
    strategy: CheckpointStrategy,
    /// 当前最佳loss
    best_loss: f32,
    /// 最佳检查点的epoch
    best_epoch: usize,
    /// 保留的最佳检查点数量
    keep_best_n: usize,
}

impl CheckpointManager {
    /// 创建新的检查点管理器
    ///
    /// # 参数
    /// - `checkpoint_dir`: 检查点保存目录
    /// - `strategy`: 保存策略
    /// - `keep_best_n`: 保留的最佳检查点数量（默认3）
    pub fn new<P: AsRef<Path>>(
        checkpoint_dir: P,
        strategy: CheckpointStrategy,
        keep_best_n: usize,
    ) -> Result<Self, String> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // 创建检查点目录
        if !checkpoint_dir.exists() {
            fs::create_dir_all(&checkpoint_dir)
                .map_err(|e| format!("创建检查点目录失败: {}", e))?;
        }

        Ok(Self {
            checkpoint_dir,
            strategy,
            best_loss: f32::INFINITY,
            best_epoch: 0,
            keep_best_n,
        })
    }

    /// 检查是否应该保存检查点
    pub fn should_save(&self, epoch: usize, current_loss: f32) -> bool {
        match &self.strategy {
            CheckpointStrategy::Best => current_loss < self.best_loss,
            CheckpointStrategy::Last => true,
            CheckpointStrategy::Periodic(n) => epoch % n == 0,
            CheckpointStrategy::BestAndLast => true,
            CheckpointStrategy::BestAndPeriodic(n) => {
                current_loss < self.best_loss || epoch % n == 0
            }
        }
    }

    /// 保存检查点
    ///
    /// # 参数
    /// - `llm`: LLM模型
    /// - `metadata`: 检查点元数据
    pub fn save_checkpoint(
        &mut self,
        llm: &LLM,
        metadata: CheckpointMetadata,
    ) -> Result<PathBuf, String> {
        let is_best = metadata.loss < self.best_loss;

        if is_best {
            self.best_loss = metadata.loss;
            self.best_epoch = metadata.epoch;
        }

        // 构建检查点文件名
        let checkpoint_name = match &self.strategy {
            CheckpointStrategy::Best => {
                if is_best {
                    format!(
                        "checkpoint_best_epoch_{}_loss_{:.4}.bin",
                        metadata.epoch, metadata.loss
                    )
                } else {
                    return Ok(PathBuf::new()); // 不保存
                }
            }
            CheckpointStrategy::Last => "checkpoint_last.bin".to_string(),
            CheckpointStrategy::Periodic(_) => {
                format!("checkpoint_epoch_{}.bin", metadata.epoch)
            }
            CheckpointStrategy::BestAndLast => {
                if is_best {
                    format!(
                        "checkpoint_best_epoch_{}_loss_{:.4}.bin",
                        metadata.epoch, metadata.loss
                    )
                } else {
                    "checkpoint_last.bin".to_string()
                }
            }
            CheckpointStrategy::BestAndPeriodic(n) => {
                if is_best {
                    format!(
                        "checkpoint_best_epoch_{}_loss_{:.4}.bin",
                        metadata.epoch, metadata.loss
                    )
                } else if metadata.epoch % n == 0 {
                    format!("checkpoint_epoch_{}.bin", metadata.epoch)
                } else {
                    return Ok(PathBuf::new()); // 不保存
                }
            }
        };

        let checkpoint_path = self.checkpoint_dir.join(&checkpoint_name);

        // 序列化模型层
        let mut serializable_layers = Vec::new();
        for layer in llm.network.iter() {
            match crate::model_serialization::SerializableLayer::from_layer(layer) {
                Ok(s_layer) => {
                    serializable_layers.push(s_layer);
                }
                Err(e) => {
                    return Err(format!("序列化层失败: {}", e));
                }
            }
        }

        let serializable_model = SerializableModel {
            version: 1,
            vocab: llm.vocab.clone(),
            layers: serializable_layers,
            context_window: llm.context_window.clone(),
            metadata: crate::model_serialization::ModelMetadata {
                embedding_dim: crate::EMBEDDING_DIM,
                hidden_dim: crate::HIDDEN_DIM,
                num_transformer_blocks: 2,
                vocab_size: llm.vocab.len(),
                max_seq_len: llm.max_context_length,
                training_info: None,
            },
        };

        let checkpoint = Checkpoint {
            model: serializable_model,
            metadata: metadata.clone(),
        };

        // 保存为二进制格式
        let file =
            fs::File::create(&checkpoint_path).map_err(|e| format!("创建检查点文件失败: {}", e))?;
        let mut writer = std::io::BufWriter::new(file);

        bincode::encode_into_std_write(&checkpoint, &mut writer, bincode::config::standard())
            .map_err(|e| format!("序列化检查点失败: {}", e))?;

        // 同时保存JSON格式的元数据（方便查看）
        let metadata_path = checkpoint_path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&metadata)
            .map_err(|e| format!("序列化元数据失败: {}", e))?;
        fs::write(&metadata_path, metadata_json).map_err(|e| format!("保存元数据失败: {}", e))?;

        log::info!(
            "📦 检查点已保存: {} (epoch={}, loss={:.4}{})",
            checkpoint_path.display(),
            metadata.epoch,
            metadata.loss,
            if is_best { ", 🏆 NEW BEST!" } else { "" }
        );

        // 清理旧的检查点
        if is_best {
            self.cleanup_old_checkpoints()?;
        }

        Ok(checkpoint_path)
    }

    /// 加载检查点
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<(LLM, CheckpointMetadata), String> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(format!("检查点文件不存在: {}", path.display()));
        }

        log::info!("📂 正在加载检查点: {}", path.display());

        let file = fs::File::open(path).map_err(|e| format!("打开检查点文件失败: {}", e))?;
        let mut reader = std::io::BufReader::new(file);

        let checkpoint: Checkpoint =
            bincode::decode_from_std_read(&mut reader, bincode::config::standard())
                .map_err(|e| format!("反序列化检查点失败: {}", e))?;

        // 重建模型
        let mut network: Vec<Box<dyn crate::llm::Layer>> = Vec::new();
        for s_layer in checkpoint.model.layers.iter() {
            let layer = s_layer.to_layer(checkpoint.model.vocab.len());
            network.push(layer);
        }

        let llm = LLM {
            vocab: checkpoint.model.vocab,
            network,
            context_window: checkpoint.model.context_window,
            max_context_length: checkpoint.model.metadata.max_seq_len,
            training: false,
            parallel_training: false,
            sampling_prob_buffer: Vec::new(),
            sampling_idx_buffer: Vec::new(),
            beam_candidates_buffer: Vec::new(),
        };

        log::info!(
            "✅ 检查点加载成功: epoch={}, loss={:.4}, phase={}",
            checkpoint.metadata.epoch,
            checkpoint.metadata.loss,
            checkpoint.metadata.phase
        );

        Ok((llm, checkpoint.metadata))
    }

    /// 获取最佳检查点路径
    pub fn get_best_checkpoint(&self) -> Option<PathBuf> {
        if self.best_loss == f32::INFINITY {
            return None;
        }

        // 查找所有best检查点文件
        let mut best_checkpoints: Vec<_> = fs::read_dir(&self.checkpoint_dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("checkpoint_best") && name_str.ends_with(".bin")
            })
            .collect();

        // 按修改时间排序，最新的在前
        best_checkpoints.sort_by_key(|entry| entry.metadata().and_then(|m| m.modified()).ok());
        best_checkpoints.reverse();

        best_checkpoints.first().map(|entry| entry.path())
    }

    /// 获取最新检查点路径
    pub fn get_last_checkpoint(&self) -> Option<PathBuf> {
        let last_path = self.checkpoint_dir.join("checkpoint_last.bin");
        if last_path.exists() {
            Some(last_path)
        } else {
            None
        }
    }

    /// 列出所有检查点
    pub fn list_checkpoints(&self) -> Result<Vec<(PathBuf, CheckpointMetadata)>, String> {
        let mut checkpoints = Vec::new();

        let entries =
            fs::read_dir(&self.checkpoint_dir).map_err(|e| format!("读取检查点目录失败: {}", e))?;

        for entry in entries {
            let entry = entry.map_err(|e| format!("读取目录项失败: {}", e))?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("bin") {
                // 读取对应的JSON元数据
                let metadata_path = path.with_extension("json");
                if metadata_path.exists() {
                    let metadata_json = fs::read_to_string(&metadata_path)
                        .map_err(|e| format!("读取元数据失败: {}", e))?;
                    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json)
                        .map_err(|e| format!("解析元数据失败: {}", e))?;
                    checkpoints.push((path, metadata));
                }
            }
        }

        // 按epoch排序
        checkpoints.sort_by_key(|(_, metadata)| metadata.epoch);

        Ok(checkpoints)
    }

    /// 清理旧的检查点，只保留最佳的N个
    fn cleanup_old_checkpoints(&self) -> Result<(), String> {
        let mut best_checkpoints: Vec<_> = fs::read_dir(&self.checkpoint_dir)
            .map_err(|e| format!("读取检查点目录失败: {}", e))?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                name_str.starts_with("checkpoint_best") && name_str.ends_with(".bin")
            })
            .filter_map(|entry| {
                let path = entry.path();
                let metadata_path = path.with_extension("json");
                if metadata_path.exists() {
                    let metadata_json = fs::read_to_string(&metadata_path).ok()?;
                    let metadata: CheckpointMetadata = serde_json::from_str(&metadata_json).ok()?;
                    Some((path, metadata))
                } else {
                    None
                }
            })
            .collect();

        // 按loss排序（最好的在前）
        best_checkpoints.sort_by(|(_, a), (_, b)| {
            a.loss
                .partial_cmp(&b.loss)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 删除多余的检查点
        for (path, metadata) in best_checkpoints.iter().skip(self.keep_best_n) {
            log::info!(
                "🗑️  删除旧检查点: {} (epoch={}, loss={:.4})",
                path.display(),
                metadata.epoch,
                metadata.loss
            );
            fs::remove_file(path).map_err(|e| format!("删除检查点失败: {}", e))?;

            // 同时删除元数据文件
            let metadata_path = path.with_extension("json");
            if metadata_path.exists() {
                fs::remove_file(metadata_path).ok();
            }
        }

        Ok(())
    }

    /// 更新最佳loss（用于EarlyStopping集成）
    pub fn update_best_loss(&mut self, loss: f32, epoch: usize) {
        if loss < self.best_loss {
            self.best_loss = loss;
            self.best_epoch = epoch;
        }
    }

    /// 获取最佳loss
    pub fn get_best_loss(&self) -> f32 {
        self.best_loss
    }

    /// 获取最佳epoch
    pub fn get_best_epoch(&self) -> usize {
        self.best_epoch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_strategy_should_save() {
        let manager =
            CheckpointManager::new("test_checkpoints", CheckpointStrategy::Periodic(10), 3)
                .unwrap();

        assert!(manager.should_save(10, 1.0));
        assert!(!manager.should_save(11, 1.0));
        assert!(manager.should_save(20, 1.0));
    }

    #[test]
    fn test_best_checkpoint_update() {
        let mut manager =
            CheckpointManager::new("test_checkpoints", CheckpointStrategy::Best, 3).unwrap();

        manager.update_best_loss(2.0, 10);
        assert_eq!(manager.get_best_loss(), 2.0);
        assert_eq!(manager.get_best_epoch(), 10);

        manager.update_best_loss(1.5, 20);
        assert_eq!(manager.get_best_loss(), 1.5);
        assert_eq!(manager.get_best_epoch(), 20);

        // 不应该更新（loss更高）
        manager.update_best_loss(2.5, 30);
        assert_eq!(manager.get_best_loss(), 1.5);
        assert_eq!(manager.get_best_epoch(), 20);
    }
}
