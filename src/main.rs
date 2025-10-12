use std::io::Write;

use ::llm::{EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQ_LEN};
use dataset_loader::{Dataset, DatasetType};

use crate::{
    embeddings::Embeddings, llm::LLM, output_projection::OutputProjection,
    transformer::TransformerBlock, vocab::Vocab,
    performance_monitor::PerformanceMonitor,
};

mod adam;
mod dataset_loader;
mod dropout;
mod embeddings;
mod feed_forward;
mod layer_norm;
mod llm;
mod output_projection;
mod position_encoding;
mod semantic_enhancer;
mod self_attention;
mod transformer;
mod vocab;
mod performance_monitor;

fn main() {
    // 创建性能监控器
    let mut perf_monitor = PerformanceMonitor::new();

    perf_monitor.start("程序总执行时间");

    // Mock input - test conversational format
    let string = String::from("用户：山脉是如何形成的？");

    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    ); // Placeholder, not used in this example
    perf_monitor.stop("加载训练数据");

    // Extract all unique words from training data to create vocabulary
    let mut vocab_set = std::collections::HashSet::new();

    // Process all training examples for vocabulary
    // First process pre-training data
    perf_monitor.start("构建词汇表 - 预训练数据");
    println!("Processing pre-training data for vocabulary...");
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    println!("Added tokens from pre-training data. Current vocabulary size: {}", vocab_set.len());
    perf_monitor.stop("构建词汇表 - 预训练数据");

    // Then process chat training data
    perf_monitor.start("构建词汇表 - 对话数据");
    println!("Processing chat training data for vocabulary...");
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);
    println!("Added tokens from chat training data. Final vocabulary size: {}", vocab_set.len());
    perf_monitor.stop("构建词汇表 - 对话数据");

    perf_monitor.start("创建词汇表对象");
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort(); // Sort for deterministic ordering
    println!("Creating vocabulary with {} unique tokens...", vocab_words.len());
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s: &String| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);
    println!("Vocabulary created successfully with {} total tokens (including special tokens)", vocab.len());
    perf_monitor.stop("创建词汇表对象");

    perf_monitor.start("初始化神经网络");
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_3 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_4 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM); // Added extra transformer block for better Chinese understanding
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());
    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(transformer_block_3),
            Box::new(transformer_block_4),
            Box::new(output_projection),
        ],
    );

    llm.set_training_mode(true);
    perf_monitor.stop("初始化神经网络");

    println!("\n=== MODEL INFORMATION ===");
    println!("Network architecture: {}", llm.network_description());
    println!(
        "Model configuration -> max_seq_len: {}, embedding_dim: {}, hidden_dim: {}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );

    println!("Total parameters: {}", llm.total_parameters());

    println!("\n=== BEFORE TRAINING ===");
    perf_monitor.start("训练前预测");
    println!("Input: {}", string);
    println!("Output: {}", llm.predict(&string));
    perf_monitor.stop("训练前预测");

    println!("\n=== PRE-TRAINING MODEL ===");
    println!(
        "Pre-training on {} examples for {} epochs with learning rate {}",
        dataset.pretraining_data.len(),
        100,
        0.0005
    );

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("预训练阶段");
    llm.train(pretraining_examples, 100, 0.0005); // Pre-training with learning rate scheduling
    perf_monitor.stop("预训练阶段");

    println!("\n=== INSTRUCTION TUNING ===");
    println!(
        "Instruction tuning on {} examples for {} epochs with learning rate {}",
        dataset.chat_training_data.len(),
        100,
        0.0001
    );

    perf_monitor.start("指令微调阶段");
    llm.train(chat_training_examples, 100, 0.0001); // Instruction tuning with learning rate scheduling
    perf_monitor.stop("指令微调阶段");

    println!("\n=== AFTER TRAINING ===");
    println!("Input: {}", string);
    llm.set_training_mode(false);

    perf_monitor.start("训练后预测 (Beam Search)");
    let result = llm.predict_with_beam_search(&string, 3, 20);
    perf_monitor.stop("训练后预测 (Beam Search)");

    println!("Output: {}", result);
    println!("======================\n");

    perf_monitor.stop("训练总消耗时间");

    // 打印性能报告
    perf_monitor.print_report();

    println!("\n--- Interactive Mode ---");
    println!("Type a prompt and press Enter to generate text.");
    println!("Type 'exit' to quit.");

    let mut input = String::new();
    loop {
        input.clear();

        print!("\nEnter prompt: ");
        std::io::stdout().flush().unwrap();

        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }

        let formatted_input = format!("User: {}", trimmed_input);
        let prediction = llm.predict_with_context(&formatted_input, 0.8, 0.9, 5);
        println!("Model output: {}", prediction);
        if prediction.contains("</s>") {
            llm.clear_context();
        }
        println!("Model output: {}", prediction);
    }
}
