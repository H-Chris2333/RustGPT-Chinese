// 模型序列化测试

use llm::{LLM, load_model_binary, load_model_json, save_model_binary, save_model_json};
use std::fs;

#[test]
fn test_binary_save_and_load() {
    // 创建测试目录
    fs::create_dir_all("test_checkpoints").unwrap();

    // 创建模型
    let llm = LLM::default();
    let original_params = llm.total_parameters();
    let original_vocab_size = llm.vocab.len();

    // 保存
    let path = "test_checkpoints/test_model.bin";
    save_model_binary(&llm, path).expect("Failed to save model");

    // 确认文件存在
    assert!(std::path::Path::new(path).exists());

    // 加载
    let loaded_llm = load_model_binary(path).expect("Failed to load model");

    // 验证
    assert_eq!(loaded_llm.total_parameters(), original_params);
    assert_eq!(loaded_llm.vocab.len(), original_vocab_size);
    assert_eq!(loaded_llm.network.len(), llm.network.len());

    // 清理
    fs::remove_file(path).unwrap();
    fs::remove_dir("test_checkpoints").unwrap();

    println!("✓ 二进制格式保存/加载测试通过!");
}

#[test]
fn test_json_save_and_load() {
    // 创建测试目录
    fs::create_dir_all("test_exports").unwrap();

    // 创建模型
    let llm = LLM::default();
    let original_params = llm.total_parameters();
    let original_vocab_size = llm.vocab.len();

    // 保存
    let path = "test_exports/test_model.json";
    save_model_json(&llm, path).expect("Failed to save model");

    // 确认文件存在
    assert!(std::path::Path::new(path).exists());

    // 加载
    let loaded_llm = load_model_json(path).expect("Failed to load model");

    // 验证
    assert_eq!(loaded_llm.total_parameters(), original_params);
    assert_eq!(loaded_llm.vocab.len(), original_vocab_size);
    assert_eq!(loaded_llm.network.len(), llm.network.len());

    // 清理
    fs::remove_file(path).unwrap();
    fs::remove_dir("test_exports").unwrap();

    println!("✓ JSON格式保存/加载测试通过!");
}

#[test]
fn test_model_state_preservation() {
    use llm::adam::Adam;
    use llm::model_serialization::SerializableAdam;

    // 测试Adam优化器状态保存
    let adam = Adam::new((10, 20));
    let serialized = SerializableAdam::from_adam(&adam);
    let deserialized = serialized.to_adam();

    assert_eq!(deserialized.timestep, adam.timestep);
    assert_eq!(deserialized.m.dim(), adam.m.dim());
    assert_eq!(deserialized.v.dim(), adam.v.dim());

    println!("✓ 优化器状态保存测试通过!");
}
