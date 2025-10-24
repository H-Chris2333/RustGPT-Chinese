//! # 性能基准测试（v0.4.0）
//!
//! 测试各项优化的性能提升：
//! 1. BLAS 加速的张量计算
//! 2. Tokenizer LRU 缓存
//! 3. KV-Cache 推理加速
//! 4. 算子融合优化
//!
//! ## 运行方式
//! ```bash
//! cargo bench --bench performance_benchmark
//! ```

use std::time::Instant;

use llm::{
    fused_ops::{FusedGELULinear, FusedLayerNormLinear},
    self_attention::SelfAttention,
    vocab::{get_cache_hit_rate, reset_cache_stats, Vocab},
    Layer, EMBEDDING_DIM,
};
use ndarray::Array2;

fn main() {
    println!("=== RustGPT-Chinese 性能基准测试 v0.4.0 ===\n");

    // 测试1: 张量计算性能（BLAS 加速）
    benchmark_tensor_operations();

    // 测试2: Tokenizer 缓存性能
    benchmark_tokenizer_cache();

    // 测试3: KV-Cache 推理加速
    benchmark_kv_cache();

    // 测试4: 算子融合性能
    benchmark_fused_ops();

    println!("\n=== 所有基准测试完成 ===");
}

/// 测试张量计算性能（BLAS 加速）
fn benchmark_tensor_operations() {
    println!("📊 测试1: 张量计算性能（BLAS 加速）");
    println!("----------------------------------------");

    let sizes = [(128, 256), (256, 512), (512, 1024)];

    for (rows, cols) in sizes {
        let a = Array2::<f32>::ones((rows, cols));
        let b = Array2::<f32>::ones((cols, rows));

        let iterations = 100;
        let start = Instant::now();

        for _ in 0..iterations {
            let _c = a.dot(&b);
        }

        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / iterations as f64;

        println!(
            "  矩阵乘法 ({} × {}) × ({} × {}): {:.2} μs/次",
            rows, cols, cols, rows, avg_time
        );
    }

    println!("  ✓ 张量计算基准测试完成\n");
}

/// 测试 Tokenizer 缓存性能
fn benchmark_tokenizer_cache() {
    println!("📊 测试2: Tokenizer 缓存性能");
    println!("----------------------------------------");

    let texts: Vec<String> = vec![
        "深度学习是人工智能的重要分支".to_string(),
        "自然语言处理技术发展迅速".to_string(),
        "Transformer模型改变了NLP领域".to_string(),
        "深度学习是人工智能的重要分支".to_string(), // 重复
        "自然语言处理技术发展迅速".to_string(),     // 重复
    ];

    let vocab = Vocab::build_from_texts(&texts);
    reset_cache_stats();

    // 第一轮：冷启动
    let start = Instant::now();
    for text in &texts {
        let _tokens = vocab.encode_sequence(text);
    }
    let cold_time = start.elapsed();

    let (hits1, misses1, rate1) = get_cache_hit_rate();
    println!("  冷启动: {} ms", cold_time.as_millis());
    println!("    - 缓存命中: {}", hits1);
    println!("    - 缓存未命中: {}", misses1);
    println!("    - 命中率: {:.1}%", rate1 * 100.0);

    // 第二轮：热缓存
    let start = Instant::now();
    for text in &texts {
        let _tokens = vocab.encode_sequence(text);
    }
    let warm_time = start.elapsed();

    let (hits2, misses2, rate2) = get_cache_hit_rate();
    println!("\n  热缓存: {} ms", warm_time.as_millis());
    println!("    - 缓存命中: {}", hits2);
    println!("    - 缓存未命中: {}", misses2);
    println!("    - 命中率: {:.1}%", rate2 * 100.0);

    let speedup = cold_time.as_micros() as f64 / warm_time.as_micros() as f64;
    println!("\n  加速比: {:.2}x", speedup);
    println!("  ✓ Tokenizer 缓存基准测试完成\n");
}

/// 测试 KV-Cache 推理加速
fn benchmark_kv_cache() {
    println!("📊 测试3: KV-Cache 推理加速");
    println!("----------------------------------------");

    let mut attention_no_cache = SelfAttention::new(EMBEDDING_DIM);
    let mut attention_with_cache = SelfAttention::new(EMBEDDING_DIM);
    attention_with_cache.enable_kv_cache();

    let sequence_lengths = [10, 20, 50];

    for seq_len in sequence_lengths {
        // 无缓存
        let input = Array2::<f32>::ones((seq_len, EMBEDDING_DIM));
        let start = Instant::now();
        for _ in 0..10 {
            let _output = attention_no_cache.forward(&input);
        }
        let time_no_cache = start.elapsed();

        // 有缓存（模拟增量生成）
        attention_with_cache.clear_kv_cache();
        let start = Instant::now();
        for _i in 0..seq_len {
            let input_single = Array2::<f32>::ones((1, EMBEDDING_DIM));
            let _output = attention_with_cache.forward_with_kv_cache(&input_single);
        }
        let time_with_cache = start.elapsed();

        let speedup = time_no_cache.as_micros() as f64 / time_with_cache.as_micros() as f64;
        println!(
            "  序列长度 {}: 无缓存 {} μs, 有缓存 {} μs, 加速比 {:.2}x",
            seq_len,
            time_no_cache.as_micros() / 10,
            time_with_cache.as_micros(),
            speedup
        );
    }

    println!("  ✓ KV-Cache 基准测试完成\n");
}

/// 测试算子融合性能
fn benchmark_fused_ops() {
    println!("📊 测试4: 算子融合性能");
    println!("----------------------------------------");

    // 测试 FusedLayerNormLinear
    {
        let mut fused_op = FusedLayerNormLinear::new(512, 1024);
        let input = Array2::<f32>::ones((32, 512));

        let start = Instant::now();
        for _ in 0..100 {
            let _output = fused_op.forward(&input);
        }
        let elapsed = start.elapsed();

        println!(
            "  FusedLayerNormLinear (32×512 → 32×1024): {:.2} μs/次",
            elapsed.as_micros() as f64 / 100.0
        );
    }

    // 测试 FusedGELULinear
    {
        let mut fused_op = FusedGELULinear::new(512, 1024);
        let input = Array2::<f32>::ones((32, 512));

        let start = Instant::now();
        for _ in 0..100 {
            let _output = fused_op.forward(&input);
        }
        let elapsed = start.elapsed();

        println!(
            "  FusedGELULinear (32×512 → 32×1024): {:.2} μs/次",
            elapsed.as_micros() as f64 / 100.0
        );
    }

    println!("  ✓ 算子融合基准测试完成\n");
}
