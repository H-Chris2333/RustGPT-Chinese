use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 性能监控工具：追踪各个操作的执行时间
pub struct PerformanceMonitor {
    timers: HashMap<String, Vec<Duration>>,
    current_timers: HashMap<String, Instant>,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            current_timers: HashMap::new(),
        }
    }

    /// 开始计时
    pub fn start(&mut self, name: &str) {
        self.current_timers.insert(name.to_string(), Instant::now());
        println!("⏱️  开始: {}", name);
    }

    /// 结束计时并记录
    pub fn stop(&mut self, name: &str) {
        if let Some(start_time) = self.current_timers.remove(name) {
            let elapsed = start_time.elapsed();
            self.timers
                .entry(name.to_string())
                .or_insert_with(Vec::new)
                .push(elapsed);

            println!("✓ 完成: {} (耗时: {:.2}秒)", name, elapsed.as_secs_f32());
        }
    }

    /// 获取某个操作的平均耗时
    #[allow(dead_code)]
    pub fn get_average(&self, name: &str) -> Option<Duration> {
        self.timers.get(name).map(|durations| {
            let total: Duration = durations.iter().sum();
            total / durations.len() as u32
        })
    }

    /// 获取某个操作的总耗时
    #[allow(dead_code)]
    pub fn get_total(&self, name: &str) -> Option<Duration> {
        self.timers
            .get(name)
            .map(|durations| durations.iter().sum())
    }

    /// 打印性能报告
    pub fn print_report(&self) {
        println!("\n╔══════════════════════════════════════════════════════════╗");
        println!("║              📊 性能监控报告                              ║");
        println!("╠══════════════════════════════════════════════════════════╣");

        let mut items: Vec<_> = self.timers.iter().collect();
        items.sort_by_key(|(name, _)| *name);

        for (name, durations) in items {
            let count = durations.len();
            let total: Duration = durations.iter().sum();
            let average = total / count as u32;

            println!("║ {:40} ║", name);
            println!("║   调用次数: {:6}                               ║", count);
            println!(
                "║   总耗时:   {:8.2}秒                          ║",
                total.as_secs_f32()
            );
            println!(
                "║   平均耗时: {:8.2}秒                          ║",
                average.as_secs_f32()
            );
            println!("╠──────────────────────────────────────────────────────────╣");
        }

        println!("╚══════════════════════════════════════════════════════════╝\n");
    }

    /// 清空所有统计数据
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.timers.clear();
        self.current_timers.clear();
    }
}

/// 便捷的作用域计时器
#[allow(dead_code)]
pub struct ScopedTimer<'a> {
    monitor: &'a mut PerformanceMonitor,
    name: String,
}

#[allow(dead_code)]
impl<'a> ScopedTimer<'a> {
    pub fn new(monitor: &'a mut PerformanceMonitor, name: &str) -> Self {
        monitor.start(name);
        Self {
            monitor,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ScopedTimer<'a> {
    fn drop(&mut self) {
        self.monitor.stop(&self.name);
    }
}
