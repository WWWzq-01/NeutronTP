#!/usr/bin/env python3
"""
测试不同收敛配置的性能改善
"""
import subprocess
import time
import sys

def run_training_config(config_name, args):
    """运行指定配置的训练"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"Args: {' '.join(args)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            "python", "dist_main_with_feedback.py",
            "--epoch=5"  # 减少epoch以快速测试
        ] + args, 
        capture_output=True, text=True, timeout=300)  # 5分钟超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 解析输出中的收敛信息
        converged_nodes = 0
        eligible_nodes = 0
        for line in result.stdout.split('\n'):
            if 'nodes converged' in line and 'eligible' in line:
                # 提取数字
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.endswith('/2238731'):
                        try:
                            converged_nodes = int(part.split('/')[0])
                        except:
                            pass
                    elif part.startswith('over') and i+1 < len(parts):
                        try:
                            eligible_nodes = int(parts[i+1])
                        except:
                            pass
        
        print(f"Duration: {duration:.1f}s")
        print(f"Converged nodes: {converged_nodes}")
        print(f"Eligible nodes: {eligible_nodes}")
        
        return {
            'config': config_name,
            'duration': duration,
            'converged_nodes': converged_nodes,
            'eligible_nodes': eligible_nodes,
            'success': result.returncode == 0,
            'stderr': result.stderr if result.stderr else ""
        }
        
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after 5 minutes")
        return {
            'config': config_name,
            'duration': 300,
            'converged_nodes': 0,
            'eligible_nodes': 0,
            'success': False,
            'stderr': "Timeout"
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'config': config_name,
            'duration': 0,
            'converged_nodes': 0,
            'eligible_nodes': 0,
            'success': False,
            'stderr': str(e)
        }

def main():
    # 测试配置
    configs = [
        {
            'name': 'Current (Random + Threshold=0.8)',
            'args': ['--enable_feedback', '--sampling_strategy=no_importance', 
                    '--similarity_threshold=0.8', '--feedback_batch_cap=20000']
        },
        {
            'name': 'Lower Threshold (0.7)',
            'args': ['--enable_feedback', '--sampling_strategy=no_importance', 
                    '--similarity_threshold=0.7', '--feedback_batch_cap=20000']
        },
        {
            'name': 'Lower Threshold (0.6)',
            'args': ['--enable_feedback', '--sampling_strategy=no_importance', 
                    '--similarity_threshold=0.6', '--feedback_batch_cap=20000']
        },
        {
            'name': 'Convergence Aware + Threshold=0.7',
            'args': ['--enable_feedback', '--sampling_strategy=convergence_aware', 
                    '--similarity_threshold=0.7', '--feedback_batch_cap=15000']
        },
        {
            'name': 'Adaptive Importance + Threshold=0.7',
            'args': ['--enable_feedback', '--sampling_strategy=adaptive_importance', 
                    '--similarity_threshold=0.7', '--feedback_batch_cap=15000']
        },
        {
            'name': 'Standard (No Feedback)',
            'args': ['--disable_feedback']
        }
    ]
    
    results = []
    
    for config in configs:
        result = run_training_config(config['name'], config['args'])
        results.append(result)
        
        # 等待一下避免系统过载
        time.sleep(10)
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")
    print(f"{'Config':<35} {'Time(s)':<8} {'Converged':<10} {'Eligible':<10} {'Status'}")
    print(f"{'-'*80}")
    
    baseline_time = None
    for result in results:
        if 'Standard' in result['config']:
            baseline_time = result['duration']
            
        status = "✅ SUCCESS" if result['success'] else f"❌ FAILED: {result['stderr'][:20]}"
        print(f"{result['config']:<35} {result['duration']:<8.1f} {result['converged_nodes']:<10} {result['eligible_nodes']:<10} {status}")
    
    if baseline_time:
        print(f"\n{'='*80}")
        print("TIME SAVINGS ANALYSIS")
        print(f"{'='*80}")
        for result in results:
            if result['success'] and 'Standard' not in result['config']:
                savings = baseline_time - result['duration']
                savings_pct = (savings / baseline_time) * 100
                print(f"{result['config']:<35}: {savings:+6.1f}s ({savings_pct:+5.1f}%)")

if __name__ == "__main__":
    main()