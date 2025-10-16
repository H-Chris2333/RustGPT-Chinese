//!  训练性能优化模块
//!
//!  包含阶段1的快速优化:
//!  1.  数据预处理缓存
//!  2.  余弦退火学习率调度
//!  3.  早停机制
//!  4.  训练监控增强

use  ndarray::{Array1,  Array2};
use  crate::utils::softmax;
use  crate::llm::LLM;

impl  LLM  {
    ///  使用预tokenize的数据进行训练（性能优化版本）
    ///
    ///  这个方法接受已经tokenize的数据，避免重复tokenization
    ///  相比train方法,在500个epoch的训练中可以节省99.8%的tokenization时间
    pub  fn  train_with_cached_tokens(
        &mut  self,
        tokenized_data:  Vec<Vec<usize>>,
        epochs:  usize,
        initial_lr:  f32,
    )  {
        self.set_training_mode(true);

        for  epoch  in  0..epochs  {
            let  decay_rate:  f32  =  0.95;
            let  decay_steps  =  10.0;
            let  current_lr  =  initial_lr  *  decay_rate.powf(epoch  as  f32  /  decay_steps);

            let  mut  total_loss  =  0.0;

            //  直接使用缓存的tokenized数据，无需重复tokenize
            for  training_row  in  &tokenized_data  {
                if  training_row.len()  <  2  {
                    continue;
                }

                //  1.  Slice  input  and  targets
                let  input_ids  =  &training_row[..training_row.len()  -  1];
                let  target_ids  =  &training_row[1..];

                //  Forward  pass
                let  mut  input:  Array2<f32>  =  Array2::zeros((1,  input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x|  x  as  f32).collect::<Array1<f32>>());

                for  layer  in  &mut  self.network  {
                    input  =  layer.forward(&input);
                }

                let  logits  =  input;
                let  probs  =  softmax(&logits);
                total_loss  +=  Self::cross_entropy_loss_step(&probs,  target_ids);

                //  Backward  pass
                let  mut  grads_output  =  Self::compute_gradients_step(&probs,  target_ids);
                Self::clip_gradients(&mut  grads_output,  5.0);

                for  layer  in  self.network.iter_mut().rev()  {
                    grads_output  =  layer.backward(&grads_output,  current_lr);
                }
            }

            println!(
                "Epoch  {}:  Loss  =  {:.4},  LR  =  {:.6}",
                epoch,
                total_loss  /  tokenized_data.len()  as  f32,
                current_lr
            );
        }

        self.set_training_mode(false);
    }

    ///  余弦退火学习率调度（带重启）
    ///
    ///  #  参数
    ///  -  `initial_lr`:  初始学习率（如  0.001）
    ///  -  `epoch`:  当前epoch
    ///  -  `total_epochs`:  总epoch数
    ///  -  `num_restarts`:  重启次数（如2表示训练分为3个周期）
    ///
    ///  #  公式
    ///  ```text
    ///  lr  =  lr_min  +  0.5  *  (lr_max  -  lr_min)  *  (1  +  cos(π  *  progress))
    ///  ```
    ///  其中  progress  =  (epoch  %  cycle_length)  /  cycle_length
    pub  fn  cosine_annealing_lr(
        initial_lr:  f32,
        epoch:  usize,
        total_epochs:  usize,
        num_restarts:  usize,
    )  ->  f32  {
        //  计算每个周期的长度
        let  cycle_length  =  total_epochs  /  (num_restarts  +  1);

        //  当前在周期内的位置
        let  cycle_epoch  =  epoch  %  cycle_length;

        //  周期内的进度  [0,  1]
        let  progress  =  cycle_epoch  as  f32  /  cycle_length  as  f32;

        //  最小学习率为初始值的1%
        let  min_lr  =  initial_lr  *  0.01;

        //  余弦退火公式
        min_lr  +  0.5  *  (initial_lr  -  min_lr)  *  (1.0  +  (std::f32::consts::PI  *  progress).cos())
    }

    ///  改进的训练方法：使用余弦退火学习率
    pub  fn  train_with_cosine_lr(
        &mut  self,
        tokenized_data:  Vec<Vec<usize>>,
        epochs:  usize,
        initial_lr:  f32,
        num_restarts:  usize,  //  推荐值:  2-3
    )  {
        self.set_training_mode(true);

        for  epoch  in  0..epochs  {
            //  🔥  使用余弦退火学习率
            let  current_lr  =  Self::cosine_annealing_lr(initial_lr,  epoch,  epochs,  num_restarts);

            let  mut  total_loss  =  0.0;
            for  training_row  in  &tokenized_data  {
                if  training_row.len()  <  2  {
                    continue;
                }

                let  input_ids  =  &training_row[..training_row.len()  -  1];
                let  target_ids  =  &training_row[1..];

                let  mut  input:  Array2<f32>  =  Array2::zeros((1,  input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x|  x  as  f32).collect::<Array1<f32>>());

                for  layer  in  &mut  self.network  {
                    input  =  layer.forward(&input);
                }

                let  logits  =  input;
                let  probs  =  softmax(&logits);
                total_loss  +=  Self::cross_entropy_loss_step(&probs,  target_ids);

                let  mut  grads_output  =  Self::compute_gradients_step(&probs,  target_ids);
                Self::clip_gradients(&mut  grads_output,  5.0);

                for  layer  in  self.network.iter_mut().rev()  {
                    grads_output  =  layer.backward(&grads_output,  current_lr);
                }
            }

            //  每10个epoch打印一次，减少输出
            if  epoch  %  10  ==  0  ||  epoch  ==  epochs  -  1  {
                println!(
                    "Epoch  {}:  Loss  =  {:.4},  LR  =  {:.6}",
                    epoch,
                    total_loss  /  tokenized_data.len()  as  f32,
                    current_lr
                );
            }
        }

        self.set_training_mode(false);
    }

    ///  带早停的训练方法
    ///
    ///  #  参数
    ///  -  `patience`:  容忍多少个epoch  loss不改善（推荐30-50）
    ///
    ///  #  返回值
    ///  返回实际训练的epoch数
    pub  fn  train_with_early_stopping(
        &mut  self,
        tokenized_data:  Vec<Vec<usize>>,
        max_epochs:  usize,
        initial_lr:  f32,
        patience:  usize,
    )  ->  usize  {
        self.set_training_mode(true);

        let  mut  best_loss  =  f32::INFINITY;
        let  mut  counter  =  0;
        let  min_delta  =  0.001f32;
        let  mut  best_epoch  =  0;

        for  epoch  in  0..max_epochs  {
            let  current_lr  =  Self::cosine_annealing_lr(initial_lr,  epoch,  max_epochs,  2);

            let  mut  total_loss  =  0.0;
            for  training_row  in  &tokenized_data  {
                if  training_row.len()  <  2  {
                    continue;
                }

                let  input_ids  =  &training_row[..training_row.len()  -  1];
                let  target_ids  =  &training_row[1..];

                let  mut  input:  Array2<f32>  =  Array2::zeros((1,  input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x|  x  as  f32).collect::<Array1<f32>>());

                for  layer  in  &mut  self.network  {
                    input  =  layer.forward(&input);
                }

                let  logits  =  input;
                let  probs  =  softmax(&logits);
                total_loss  +=  Self::cross_entropy_loss_step(&probs,  target_ids);

                let  mut  grads_output  =  Self::compute_gradients_step(&probs,  target_ids);
                Self::clip_gradients(&mut  grads_output,  5.0);

                for  layer  in  self.network.iter_mut().rev()  {
                    grads_output  =  layer.backward(&grads_output,  current_lr);
                }
            }

            let  avg_loss  =  total_loss  /  tokenized_data.len()  as  f32;

            if  epoch  %  10  ==  0  ||  epoch  ==  max_epochs  -  1  {
                println!(
                    "Epoch  {}:  Loss  =  {:.4},  LR  =  {:.6}",
                    epoch,  avg_loss,  current_lr
                );
            }

            //  🔥  检查早停条件
            if  avg_loss  <  best_loss  -  min_delta  {
                best_loss  =  avg_loss;
                best_epoch  =  epoch;
                counter  =  0;
            }  else  {
                counter  +=  1;
                if  counter  >=  patience  {
                    println!("\n🛑  早停触发:");
                    println!("        •  最佳epoch:  {}",  best_epoch);
                    println!("        •  最佳loss:  {:.4}",  best_loss);
                    println!("        •  停止epoch:  {}",  epoch);
                    println!("        •  节省时间:  {}  epochs\n",  max_epochs  -  epoch);

                    self.set_training_mode(false);
                    return  epoch  +  1;
                }
            }
        }

        self.set_training_mode(false);
        max_epochs
    }

    ///  计算梯度L2范数
    fn  compute_grad_norm(grads:  &Array2<f32>)  ->  f32  {
        grads.iter().map(|&x|  x  *  x).sum::<f32>().sqrt()
    }

    ///  带完整监控的训练方法（结合早停、余弦学习率、详细统计）
    ///
    ///  这是最完整的训练方法，推荐使用
    pub  fn  train_monitored(
        &mut  self,
        tokenized_data:  Vec<Vec<usize>>,
        max_epochs:  usize,
        initial_lr:  f32,
        patience:  usize,
    )  ->  usize  {
        self.set_training_mode(true);

        let  mut  best_loss  =  f32::INFINITY;
        let  mut  counter  =  0;
        let  min_delta  =  0.001f32;
        let  mut  best_epoch  =  0;
        let  start_time  =  std::time::Instant::now();

        for  epoch  in  0..max_epochs  {
            let  epoch_start  =  std::time::Instant::now();
            let  current_lr  =  Self::cosine_annealing_lr(initial_lr,  epoch,  max_epochs,  2);

            let  mut  total_loss  =  0.0;
            let  mut  total_grad_norm  =  0.0;
            let  mut  sample_count  =  0;

            for  training_row  in  &tokenized_data  {
                if  training_row.len()  <  2  {
                    continue;
                }

                let  input_ids  =  &training_row[..training_row.len()  -  1];
                let  target_ids  =  &training_row[1..];

                let  mut  input:  Array2<f32>  =  Array2::zeros((1,  input_ids.len()));
                input
                    .row_mut(0)
                    .assign(&input_ids.iter().map(|&x|  x  as  f32).collect::<Array1<f32>>());

                for  layer  in  &mut  self.network  {
                    input  =  layer.forward(&input);
                }

                let  logits  =  input;
                let  probs  =  softmax(&logits);
                total_loss  +=  Self::cross_entropy_loss_step(&probs,  target_ids);

                let  mut  grads_output  =  Self::compute_gradients_step(&probs,  target_ids);

                //  记录梯度范数
                total_grad_norm  +=  Self::compute_grad_norm(&grads_output);

                Self::clip_gradients(&mut  grads_output,  5.0);

                for  layer  in  self.network.iter_mut().rev()  {
                    grads_output  =  layer.backward(&grads_output,  current_lr);
                }

                sample_count  +=  1;
            }

            let  epoch_time  =  epoch_start.elapsed().as_secs_f32();
            let  avg_loss  =  total_loss  /  sample_count  as  f32;
            let  avg_grad_norm  =  total_grad_norm  /  sample_count  as  f32;
            let  perplexity  =  avg_loss.exp();
            let  samples_per_sec  =  sample_count  as  f32  /  epoch_time;

            //  📊  丰富的训练信息
            if  epoch  %  10  ==  0  ||  epoch  ==  max_epochs  -  1  {
                let  progress  =  (epoch  +  1)  as  f32  /  max_epochs  as  f32  *  100.0;
                let  elapsed  =  start_time.elapsed().as_secs();
                let  eta  =  (elapsed  as  f32  /  (epoch  +  1)  as  f32  *  (max_epochs  -  epoch  -  1)  as  f32)  as  u64;

                println!(
                    "[{:3}/{:3}]  ({:.1}%)  Loss:  {:.4}  |  PPL:  {:.2}  |  LR:  {:.6}  |  Grad:  {:.4}  |  Speed:  {:.1}  samples/s  |  ETA:  {}s",
                    epoch  +  1,
                    max_epochs,
                    progress,
                    avg_loss,
                    perplexity,
                    current_lr,
                    avg_grad_norm,
                    samples_per_sec,
                    eta
                );
            }

            //  🔥  检查早停条件
            if  avg_loss  <  best_loss  -  min_delta  {
                best_loss  =  avg_loss;
                best_epoch  =  epoch;
                counter  =  0;
            }  else  {
                counter  +=  1;
                if  counter  >=  patience  {
                    println!("\n🛑  早停触发:");
                    println!("        •  最佳epoch:  {}",  best_epoch);
                    println!("        •  最佳loss:  {:.4}",  best_loss);
                    println!("        •  停止epoch:  {}",  epoch);
                    println!("        •  节省时间:  {}  epochs\n",  max_epochs  -  epoch);

                    self.set_training_mode(false);
                    return  epoch  +  1;
                }
            }
        }

        self.set_training_mode(false);
        max_epochs
    }
}
