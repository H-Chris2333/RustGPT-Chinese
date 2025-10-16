use std::io::Write;

// 从lib.rs导入所有需要的类型和常量
use llm::{
    Dataset, DatasetType, EMBEDDING_DIM, Embeddings, HIDDEN_DIM, LLM, MAX_SEQ_LEN,
    OutputProjection, PerformanceMonitor, TransformerBlock, Vocab, load_model_binary,
    save_model_binary, save_model_json,
};

// 🔥 导入训练性能优化模块

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║          RustGPT-Chinese - 中文GPT模型训练系统            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // 创建性能监控器
    let mut perf_monitor = PerformanceMonitor::new();
    perf_monitor.start("程序总执行时间");

    // 检查是否存在已保存的模型
    let model_path = "checkpoints/model_final.bin";
    let pretrain_checkpoint = "checkpoints/model_pretrained.bin";

    let mut llm = if std::path::Path::new(model_path).exists()
        || std::path::Path::new(pretrain_checkpoint).exists()
    {
        println!("🔍 检测到已保存的模型:");
        if std::path::Path::new(model_path).exists() {
            println!("   ✓ {}", model_path);
        }
        if std::path::Path::new(pretrain_checkpoint).exists() {
            println!("   ✓ {}", pretrain_checkpoint);
        }
        println!();

        print!("是否加载已有模型? (y/n): ");
        std::io::stdout().flush().unwrap();

        let mut choice = String::new();
        std::io::stdin().read_line(&mut choice).unwrap();

        if choice.trim().eq_ignore_ascii_case("y") {
            // 选择加载哪个模型
            let load_path = if std::path::Path::new(model_path).exists() {
                print!(
                    "\n选择要加载的模型:\n   1) {} (最终模型)\n   2) {} (预训练checkpoint)\n请选择 (1/2): ",
                    model_path, pretrain_checkpoint
                );
                std::io::stdout().flush().unwrap();

                let mut model_choice = String::new();
                std::io::stdin().read_line(&mut model_choice).unwrap();

                if model_choice.trim() == "2" && std::path::Path::new(pretrain_checkpoint).exists()
                {
                    pretrain_checkpoint
                } else {
                    model_path
                }
            } else {
                pretrain_checkpoint
            };

            println!("\n📂 正在加载模型: {}...", load_path);
            perf_monitor.start("加载模型");

            match load_model_binary(load_path) {
                Ok(mut loaded_llm) => {
                    perf_monitor.stop("加载模型");
                    loaded_llm.set_training_mode(false);

                    println!("\n✅ 模型加载成功!");
                    println!("   • 词汇量: {}", loaded_llm.vocab.len());
                    println!("   • 总参数: {}", loaded_llm.total_parameters());
                    println!("   • 网络架构: {}", loaded_llm.network_description());

                    // 询问是否继续训练
                    print!("\n是否继续训练此模型? (y/n): ");
                    std::io::stdout().flush().unwrap();

                    let mut train_choice = String::new();
                    std::io::stdin().read_line(&mut train_choice).unwrap();

                    if train_choice.trim().eq_ignore_ascii_case("y") {
                        continue_training_loaded_model(loaded_llm, &mut perf_monitor)
                    } else {
                        println!("\n✓ 跳过训练，直接进入交互模式");
                        loaded_llm
                    }
                }
                Err(e) => {
                    println!("\n❌ 加载模型失败: {}", e);
                    println!("将重新训练模型...\n");
                    train_new_model(&mut perf_monitor)
                }
            }
        } else {
            println!("\n🔄 将训练新模型...\n");
            train_new_model(&mut perf_monitor)
        }
    } else {
        println!("📝 未检测到已保存的模型，将开始训练新模型...\n");
        train_new_model(&mut perf_monitor)
    };

    // 训练完成后询问是否保存
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                    模型保存选项                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    print!("是否保存当前模型? (y/n): ");
    std::io::stdout().flush().unwrap();

    let mut save_choice = String::new();
    std::io::stdin().read_line(&mut save_choice).unwrap();

    if save_choice.trim().eq_ignore_ascii_case("y") {
        save_model_interactive(&llm);
    } else {
        println!("✓ 跳过保存");
    }

    // 测试模型
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      模型测试                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    let test_input = String::from("用户：山脉是如何形成的？");
    println!("测试输入: {}", test_input);

    llm.set_training_mode(false);
    perf_monitor.start("测试预测 (Beam Search)");
    let result = llm.predict_with_beam_search(&test_input, 3, 20);
    perf_monitor.stop("测试预测 (Beam Search)");

    println!("模型输出: {}", result);

    perf_monitor.stop("程序总执行时间");

    // 打印性能报告
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      性能报告                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");
    perf_monitor.print_report();

    // 进入交互模式
    interactive_mode(&mut llm);
}

/// 训练新模型（使用性能优化）
fn train_new_model(perf_monitor: &mut PerformanceMonitor) -> LLM {
    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    );
    perf_monitor.stop("加载训练数据");

    // 构建词汇表
    let mut vocab_set = std::collections::HashSet::new();

    perf_monitor.start("构建词汇表 - 预训练数据");
    println!("📝 处理预训练数据构建词汇表...");
    Vocab::process_text_for_vocab(&dataset.pretraining_data, &mut vocab_set);
    println!("✓ 预训练数据处理完成，当前词汇量: {}", vocab_set.len());
    perf_monitor.stop("构建词汇表 - 预训练数据");

    perf_monitor.start("构建词汇表 - 对话数据");
    println!("📝 处理对话数据构建词汇表...");
    Vocab::process_text_for_vocab(&dataset.chat_training_data, &mut vocab_set);
    println!("✓ 对话数据处理完成，最终词汇量: {}", vocab_set.len());
    perf_monitor.stop("构建词汇表 - 对话数据");

    perf_monitor.start("创建词汇表对象");
    let mut vocab_words: Vec<String> = vocab_set.into_iter().collect();
    vocab_words.sort();
    println!("📚 创建词汇表，共 {} 个唯一词元...", vocab_words.len());
    let vocab_words_refs: Vec<&str> = vocab_words.iter().map(|s| s.as_str()).collect();
    let vocab = Vocab::new(vocab_words_refs);
    println!("✓ 词汇表创建成功，总计 {} 个词元 (含特殊词元)", vocab.len());
    perf_monitor.stop("创建词汇表对象");

    perf_monitor.start("初始化神经网络");
    println!("\n🏗️  初始化神经网络...");
    let transformer_block_1 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let transformer_block_2 = TransformerBlock::new(EMBEDDING_DIM, HIDDEN_DIM);
    let output_projection = OutputProjection::new(EMBEDDING_DIM, vocab.words.len());
    let embeddings = Embeddings::new(vocab.clone());

    let mut llm = LLM::new(
        vocab,
        vec![
            Box::new(embeddings),
            Box::new(transformer_block_1),
            Box::new(transformer_block_2),
            Box::new(output_projection),
        ],
    );

    perf_monitor.stop("初始化神经网络");

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      模型信息                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 网络架构: {}", llm.network_description());
    println!(
        "   • 配置: max_seq_len={}, embedding_dim={}, hidden_dim={}",
        MAX_SEQ_LEN, EMBEDDING_DIM, HIDDEN_DIM
    );
    println!("   • 总参数量: {}", llm.total_parameters());

    // 训练前测试
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                  训练前模型测试                           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    let test_input = String::from("用户：山脉是如何形成的？");
    println!("\n测试输入: {}", test_input);

    llm.set_training_mode(false);
    perf_monitor.start("训练前预测");
    let before_output = llm.predict_with_beam_search(&test_input, 3, 20);
    perf_monitor.stop("训练前预测");

    println!("训练前输出: {}\n", before_output);

    // 🔥 阶段1：预训练（使用优化的训练方法）
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║            阶段1: 预训练 (Pre-training) - 优化版          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 训练样本: {}", dataset.pretraining_data.len());
    println!("   • 最大epochs: 500 (早停patience=30)");
    println!("   • 学习率: 0.001 (余弦退火, 2次重启)");
    println!("   • 梯度累积: 4步 (等效batch_size=4)");
    println!("   • 优化: 数据缓存 + 余弦退火 + 早停 + 梯度累积\n");

    let pretraining_examples: Vec<&str> = dataset
        .pretraining_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("预训练阶段");
    let actual_epochs = llm.train_monitored(
        pretraining_examples,
        500,   // max_epochs
        0.001, // initial_lr
        30,    // patience (早停容忍30个epoch)
        4,     // accumulation_steps (梯度累积4步)
    );
    perf_monitor.stop("预训练阶段");

    println!("✓ 预训练完成，实际训练 {} epochs", actual_epochs);

    // 询问是否保存预训练checkpoint
    print!("\n💾 是否保存预训练checkpoint? (y/n): ");
    std::io::stdout().flush().unwrap();

    let mut checkpoint_choice = String::new();
    std::io::stdin().read_line(&mut checkpoint_choice).unwrap();

    if checkpoint_choice.trim().eq_ignore_ascii_case("y") {
        std::fs::create_dir_all("checkpoints").ok();
        match save_model_binary(&llm, "checkpoints/model_pretrained.bin") {
            Ok(_) => println!("✓ 预训练checkpoint已保存"),
            Err(e) => println!("❌ 保存失败: {}", e),
        }
    }

    // 🔥 阶段2：指令微调（使用优化的训练方法）
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║        阶段2: 指令微调 (Instruction Tuning) - 优化版     ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("   • 训练样本: {}", dataset.chat_training_data.len());
    println!("   • 最大epochs: 500 (早停patience=30)");
    println!("   • 学习率: 0.0005 (余弦退火, 2次重启)");
    println!("   • 梯度累积: 4步 (等效batch_size=4)\n");

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    perf_monitor.start("指令微调阶段");
    let actual_epochs = llm.train_monitored(chat_training_examples, 500, 0.0005, 30, 4);
    perf_monitor.stop("指令微调阶段");

    println!("✓ 指令微调完成，实际训练 {} epochs", actual_epochs);

    println!("\n✅ 训练完成!");

    llm
}

/// 继续训练已加载的模型
fn continue_training_loaded_model(mut llm: LLM, perf_monitor: &mut PerformanceMonitor) -> LLM {
    println!("\n🔄 继续训练模式");

    // 加载数据
    perf_monitor.start("加载训练数据");
    let dataset = Dataset::new(
        String::from("data/pretraining_data.json"),
        String::from("data/chat_training_data.json"),
        DatasetType::JSON,
    );
    perf_monitor.stop("加载训练数据");

    // 询问训练参数
    print!("\n训练轮数 (默认50): ");
    std::io::stdout().flush().unwrap();
    let mut epochs_input = String::new();
    std::io::stdin().read_line(&mut epochs_input).unwrap();
    let epochs: usize = epochs_input.trim().parse().unwrap_or(50);

    print!("学习率 (默认0.0001): ");
    std::io::stdout().flush().unwrap();
    let mut lr_input = String::new();
    std::io::stdin().read_line(&mut lr_input).unwrap();
    let lr: f32 = lr_input.trim().parse().unwrap_or(0.0001);

    println!("\n开始继续训练 ({} epochs, lr={})...\n", epochs, lr);

    let chat_training_examples: Vec<&str> = dataset
        .chat_training_data
        .iter()
        .map(|s| s.as_str())
        .collect();

    llm.set_training_mode(true);
    perf_monitor.start("继续训练");

    // 🔥 使用优化的训练方法
    llm.train_monitored(
        chat_training_examples,
        epochs,
        lr,
        20, // patience (早停容忍20个epoch)
        4,  // accumulation_steps
    );

    perf_monitor.stop("继续训练");

    println!("\n✅ 继续训练完成!");

    llm
}

/// 交互式保存模型
fn save_model_interactive(llm: &LLM) {
    println!("\n选择保存格式:");
    println!("   1) 二进制格式 (.bin) - 推荐，文件小、速度快");
    println!("   2) JSON格式 (.json) - 人类可读，便于调试");
    println!("   3) 两种格式都保存");

    print!("\n请选择 (1/2/3): ");
    std::io::stdout().flush().unwrap();

    let mut format_choice = String::new();
    std::io::stdin().read_line(&mut format_choice).unwrap();

    std::fs::create_dir_all("checkpoints").ok();
    std::fs::create_dir_all("exports").ok();

    match format_choice.trim() {
        "1" => {
            print!("文件名 (默认: checkpoints/model_final.bin): ");
            std::io::stdout().flush().unwrap();

            let mut filename = String::new();
            std::io::stdin().read_line(&mut filename).unwrap();
            let path = if filename.trim().is_empty() {
                "checkpoints/model_final.bin"
            } else {
                filename.trim()
            };

            match save_model_binary(llm, path) {
                Ok(_) => println!("✅ 模型已保存: {}", path),
                Err(e) => println!("❌ 保存失败: {}", e),
            }
        }
        "2" => {
            print!("文件名 (默认: exports/model_final.json): ");
            std::io::stdout().flush().unwrap();

            let mut filename = String::new();
            std::io::stdin().read_line(&mut filename).unwrap();
            let path = if filename.trim().is_empty() {
                "exports/model_final.json"
            } else {
                filename.trim()
            };

            match save_model_json(llm, path) {
                Ok(_) => println!("✅ 模型已保存: {}", path),
                Err(e) => println!("❌ 保存失败: {}", e),
            }
        }
        "3" => {
            println!("\n保存二进制格式...");
            match save_model_binary(llm, "checkpoints/model_final.bin") {
                Ok(_) => println!("✓ 二进制格式已保存: checkpoints/model_final.bin"),
                Err(e) => println!("✗ 二进制保存失败: {}", e),
            }

            println!("保存JSON格式...");
            match save_model_json(llm, "exports/model_final.json") {
                Ok(_) => println!("✓ JSON格式已保存: exports/model_final.json"),
                Err(e) => println!("✗ JSON保存失败: {}", e),
            }
        }
        _ => println!("❌ 无效选项，跳过保存"),
    }
}

/// 交互模式
fn interactive_mode(llm: &mut LLM) {
    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                      交互模式                             ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n💡 输入问题后按回车生成回答");
    println!("💡 输入 'exit' 退出程序");
    println!("💡 输入 'clear' 清空对话上下文");
    println!("💡 输入 'save' 保存当前模型");
    println!("💡 使用KV缓存加速推理（约10-100倍）\n");

    // 启用KV缓存加速推理
    llm.enable_kv_cache();

    let mut input = String::new();
    loop {
        input.clear();

        print!("👤 用户: ");
        std::io::stdout().flush().unwrap();

        std::io::stdin()
            .read_line(&mut input)
            .expect("Failed to read input");

        let trimmed_input = input.trim();

        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("👋 感谢使用，再见!");
            break;
        }

        if trimmed_input.eq_ignore_ascii_case("clear") {
            llm.clear_context();
            llm.clear_kv_cache(); // 同时清空KV缓存
            println!("✓ 对话上下文和KV缓存已清空\n");
            continue;
        }

        if trimmed_input.eq_ignore_ascii_case("save") {
            save_model_interactive(llm);
            println!();
            continue;
        }

        let formatted_input = format!("用户：{}", trimmed_input);
        print!("🤖 模型: ");
        std::io::stdout().flush().unwrap();

        let prediction = llm.predict_with_context(&formatted_input, 0.8, 0.9, 5);
        println!("{}\n", prediction);

        if prediction.contains("</s>") {
            llm.clear_context();
        }
    }
}
