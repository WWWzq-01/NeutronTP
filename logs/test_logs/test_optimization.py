#!/usr/bin/env python3

"""
测试收敛节点优化的性能提升脚本
"""

import subprocess
import sys
import time

def run_test(test_name, command, log_file):
    """运行测试并记录时间"""
    print(f"\n=== Running {test_name} ===")
    print(f"Command: {command}")
    print(f"Log file: {log_file}")
    
    start_time = time.time()
    
    try:
        # 运行命令并将输出重定向到日志文件
        result = subprocess.run(
            command.split(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            timeout=600  # 10分钟超时
        )
        
        # 写入日志文件
        with open(log_file, 'w') as f:
            f.write(result.stdout)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ {test_name} completed in {duration:.2f} seconds")
        print(f"Return code: {result.returncode}")
        
        return duration, result.returncode == 0
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        duration = end_time - start_time
        print(f"⏰ {test_name} timed out after {duration:.2f} seconds")
        return duration, False
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ {test_name} failed with error: {e}")
        return duration, False

def analyze_logs(optimized_log, baseline_log):
    """分析日志文件，提取关键性能指标"""
    
    def extract_metrics(log_file):
        """从日志中提取指标"""
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # 提取总时间
            lines = content.strip().split('\n')
            total_time = None
            epoch_count = 0
            early_stop_epoch = None
            
            for line in lines:
                if 'total' in line and 's' in line:
                    try:
                        # 例如: "320.33s   0.01s     1 total"
                        parts = line.split()
                        if parts and 's' in parts[0]:
                            total_time = float(parts[0].replace('s', ''))
                    except:
                        continue
                
                if 'Epoch' in line and 'Loss' in line:
                    epoch_count += 1
                
                if 'early stop' in line.lower():
                    try:
                        # 提取早停的epoch数
                        if 'Epoch' in line:
                            epoch_part = line.split('Epoch')[1].split('|')[0].strip()
                            early_stop_epoch = int(epoch_part)
                    except:
                        continue
            
            return {
                'total_time': total_time,
                'epoch_count': epoch_count,
                'early_stop_epoch': early_stop_epoch
            }
            
        except Exception as e:
            print(f"Error analyzing {log_file}: {e}")
            return {}
    
    print("\n=== Performance Analysis ===")
    
    optimized_metrics = extract_metrics(optimized_log)
    baseline_metrics = extract_metrics(baseline_log)
    
    print(f"Optimized version metrics: {optimized_metrics}")
    print(f"Baseline version metrics: {baseline_metrics}")
    
    # 计算性能提升
    if optimized_metrics.get('total_time') and baseline_metrics.get('total_time'):
        time_improvement = (baseline_metrics['total_time'] - optimized_metrics['total_time']) / baseline_metrics['total_time'] * 100
        print(f"Time improvement: {time_improvement:.2f}%")
        
        if time_improvement > 0:
            print(f"🚀 Optimization successful! {time_improvement:.2f}% faster")
        else:
            print(f"⚠️  No significant improvement: {abs(time_improvement):.2f}% slower")
    
    return optimized_metrics, baseline_metrics

def main():
    """主测试函数"""
    
    # 测试配置
    base_cmd = "python dist_main_with_feedback.py --nnodes 1 --nprocs 1 --epoch 10"
    
    # 测试1: 优化版本（启用收敛节点优化）
    optimized_cmd = f"{base_cmd} --enable_feedback --sampling_strategy convergence_aware --batch_size 5000 --similarity_threshold 0.70"
    optimized_log = "test_optimized.log"
    
    # 测试2: 基线版本（禁用反馈系统）
    baseline_cmd = f"{base_cmd}"
    baseline_log = "test_baseline.log"
    
    print("Testing InfoNTP Convergence Node Optimization")
    print("=" * 50)
    
    # 运行优化版本测试
    opt_time, opt_success = run_test("Optimized Version", optimized_cmd, optimized_log)
    
    # 运行基线版本测试
    base_time, base_success = run_test("Baseline Version", baseline_cmd, baseline_log)
    
    # 分析结果
    if opt_success and base_success:
        analyze_logs(optimized_log, baseline_log)
    else:
        print("\n❌ One or both tests failed, cannot compare performance")
    
    print(f"\nTest completed!")
    print(f"Optimized log: {optimized_log}")
    print(f"Baseline log: {baseline_log}")

if __name__ == "__main__":
    main()