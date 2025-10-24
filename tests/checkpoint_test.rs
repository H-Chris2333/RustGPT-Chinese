//! 检查点管理器集成测试
//!
//! 测试检查点的保存、加载和训练恢复功能

use llm::{
    CheckpointManager, CheckpointMetadata, CheckpointStrategy, Embeddings, LLM, OutputProjection,
    TransformerBlock, Vocab, EMBEDDING_DIM, HIDDEN_DIM,
};
use std::fs;

/// 创建一个小型测试模型
fn create_test_model() -> (LLM, Vec<String>) {
    // 准备测试数据
    let test_data = vec![
        "今天天气很好".to_string(),
        "我喜欢学习编程".to_string(),
        "机器学习很有趣".to_string(),
    ];

    // 构建词汇表
    let mut vocab_set = std::collections::HashSet::new();
    Vocab::process_text_for_vocab(&test_data, &mut vocab_set);

    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);

    // 创建模型
    let embeddings = Embeddings::new(vocab.clone());
    let transformer = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());

    let llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer),
            Box::new(output_projection),
        ],
    );

    (llm, test_data)
}

#[test]
fn test_checkpoint_manager_creation() {
    let checkpoint_dir = "test_checkpoints_creation";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3);
    assert!(manager.is_ok(), "应该能成功创建检查点管理器");

    // 验证目录被创建
    assert!(
        std::path::Path::new(checkpoint_dir).exists(),
        "检查点目录应该被创建"
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_save_and_load() {
    let checkpoint_dir = "test_checkpoints_save_load";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能创建管理器");

    // 创建测试模型
    let (llm, _test_data) = create_test_model();

    // 保存检查点
    let metadata = CheckpointMetadata {
        epoch: 10,
        loss: 2.5,
        learning_rate: 0.001,
        timestamp: "2025-10-24 10:00:00".to_string(),
        phase: "test".to_string(),
    };

    let result = manager.save_checkpoint(&llm, metadata.clone());
    assert!(result.is_ok(), "应该能成功保存检查点");

    let checkpoint_path = result.unwrap();
    assert!(checkpoint_path.exists(), "检查点文件应该存在");

    // 加载检查点
    let load_result = CheckpointManager::load_checkpoint(&checkpoint_path);
    assert!(load_result.is_ok(), "应该能成功加载检查点");

    let (loaded_llm, loaded_metadata) = load_result.unwrap();

    // 验证元数据
    assert_eq!(loaded_metadata.epoch, 10, "Epoch应该匹配");
    assert_eq!(loaded_metadata.loss, 2.5, "Loss应该匹配");
    assert_eq!(loaded_metadata.phase, "test", "Phase应该匹配");

    // 验证模型结构
    assert_eq!(
        loaded_llm.vocab.len(),
        llm.vocab.len(),
        "词汇表大小应该匹配"
    );
    assert_eq!(
        loaded_llm.total_parameters(),
        llm.total_parameters(),
        "参数数量应该匹配"
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_best_loss_tracking() {
    let checkpoint_dir = "test_checkpoints_best_loss";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Best, 3)
        .expect("应该能创建管理器");

    let (llm, _test_data) = create_test_model();

    // 保存第一个检查点
    let metadata1 = CheckpointMetadata {
        epoch: 10,
        loss: 3.0,
        learning_rate: 0.001,
        timestamp: "2025-10-24 10:00:00".to_string(),
        phase: "test".to_string(),
    };
    manager.save_checkpoint(&llm, metadata1).ok();
    assert_eq!(manager.get_best_loss(), 3.0, "应该更新最佳loss");

    // 保存第二个更好的检查点
    let metadata2 = CheckpointMetadata {
        epoch: 20,
        loss: 2.5,
        learning_rate: 0.001,
        timestamp: "2025-10-24 11:00:00".to_string(),
        phase: "test".to_string(),
    };
    manager.save_checkpoint(&llm, metadata2).ok();
    assert_eq!(manager.get_best_loss(), 2.5, "应该更新为更好的loss");

    // 验证get_best_checkpoint返回最佳检查点
    let best_checkpoint = manager.get_best_checkpoint();
    assert!(best_checkpoint.is_some(), "应该有最佳检查点");

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_training_continuity() {
    let checkpoint_dir = "test_checkpoints_continuity";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
        .expect("应该能创建管理器");

    // 创建模型并训练几个epoch
    let (mut llm, test_data) = create_test_model();

    // Tokenize数据
    let tokenized_data: Vec<Vec<usize>> = test_data
        .iter()
        .map(|text| LLM::tokenize_with_vocab(&llm.vocab, text))
        .collect();

    // 训练5个epoch并保存检查点
    let epochs_trained = llm.train_with_checkpointing(
        tokenized_data.clone(),
        5,
        0.001,
        100, // 高patience确保不会早停
        Some(&mut manager),
        "test_phase",
        0,
    );

    println!("训练了 {} 个epoch", epochs_trained);

    // 列出所有保存的检查点
    println!("检查点目录内容:");
    if let Ok(entries) = fs::read_dir(checkpoint_dir) {
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  - {:?}", entry.file_name());
            }
        }
    }

    // 获取训练后的loss
    llm.set_training_mode(true);
    let mut final_loss = 0.0;
    let mut count = 0;
    for training_row in &tokenized_data {
        if training_row.len() < 2 {
            continue;
        }
        let input_ids = &training_row[..training_row.len() - 1];
        let target_ids = &training_row[1..];

        let input =
            ndarray::Array2::from_shape_fn((1, input_ids.len()), |(_, j)| input_ids[j] as f32);

        let mut output = input.clone();
        for layer in &mut llm.network {
            output = layer.forward(&output);
        }

        let probs = llm::utils::softmax(&output);
        final_loss += LLM::cross_entropy_loss_step(&probs, target_ids);
        count += 1;
    }
    final_loss /= count as f32;
    llm.set_training_mode(false);

    // 加载检查点（尝试best，如果没有则用last）
    let checkpoint_path = manager
        .get_best_checkpoint()
        .or_else(|| manager.get_last_checkpoint());
    assert!(checkpoint_path.is_some(), "应该有保存的检查点");

    let (loaded_llm, _metadata) =
        CheckpointManager::load_checkpoint(checkpoint_path.unwrap()).expect("应该能加载检查点");

    // 验证加载的模型参数数量匹配
    assert_eq!(
        loaded_llm.total_parameters(),
        llm.total_parameters(),
        "参数数量应该匹配"
    );

    // 验证词汇表匹配
    assert_eq!(
        loaded_llm.vocab.len(),
        llm.vocab.len(),
        "词汇表大小应该匹配"
    );

    println!("✓ 检查点保存和加载测试通过");
    println!("  训练后loss: {:.4}", final_loss);
    println!("  模型参数数: {}", loaded_llm.total_parameters());

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_strategy_periodic() {
    let checkpoint_dir = "test_checkpoints_periodic";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::Periodic(10), 3)
        .expect("应该能创建管理器");

    // 测试周期性保存逻辑
    assert!(manager.should_save(10, 2.0), "Epoch 10应该保存");
    assert!(!manager.should_save(11, 2.0), "Epoch 11不应该保存");
    assert!(manager.should_save(20, 2.0), "Epoch 20应该保存");
    assert!(manager.should_save(0, 2.0), "Epoch 0应该保存");

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}

#[test]
fn test_checkpoint_list_functionality() {
    let checkpoint_dir = "test_checkpoints_list";

    // 清理之前的测试数据
    if std::path::Path::new(checkpoint_dir).exists() {
        fs::remove_dir_all(checkpoint_dir).ok();
    }

    let mut manager = CheckpointManager::new(checkpoint_dir, CheckpointStrategy::BestAndLast, 3)
        .expect("应该能创建管理器");

    let (llm, _test_data) = create_test_model();

    // 保存多个检查点
    for i in 0..3 {
        let metadata = CheckpointMetadata {
            epoch: i * 10,
            loss: 3.0 - i as f32 * 0.5,
            learning_rate: 0.001,
            timestamp: format!("2025-10-24 1{}:00:00", i),
            phase: "test".to_string(),
        };
        manager.save_checkpoint(&llm, metadata).ok();
    }

    // 列出所有检查点
    let checkpoints = manager.list_checkpoints();
    assert!(checkpoints.is_ok(), "应该能列出检查点");

    let checkpoint_list = checkpoints.unwrap();
    assert!(
        checkpoint_list.len() >= 2,
        "应该至少有2个检查点（best + last）"
    );

    // 清理
    fs::remove_dir_all(checkpoint_dir).ok();
}
