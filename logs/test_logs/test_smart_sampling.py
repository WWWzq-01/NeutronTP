#!/usr/bin/env python3
"""
快速测试智能采样效果
"""
import subprocess
import time
import re

def extract_training_stats(log_content):
    """提取训练统计信息"""
    stats = {
        'converged_nodes': [],
        'eligible_nodes': [],
        'total_time': 0,
        'epoch_times': [],
        'final_converged': 0
    }
    
    # 提取每个epoch的收敛信息
    epoch_pattern = r'Epoch (\d+): (\d+)/\d+ nodes converged.*?over (\d+) nodes'
    matches = re.findall(epoch_pattern, log_content)
    
    for match in matches:
        epoch, converged, eligible = map(int, match)
        stats['converged_nodes'].append(converged)
        stats['eligible_nodes'].append(eligible)
    
    # 提取总训练时间
    time_pattern = r'(\d+\.\d+)s.*?1\s+total'
    time_match = re.search(time_pattern, log_content)
    if time_match:
        stats['total_time'] = float(time_match.group(1))
    
    # 最终收敛节点数
    if stats['converged_nodes']:
        stats['final_converged'] = stats['converged_nodes'][-1]
    
    return stats

def run_test_config(name, args, timeout=300):
    """运行测试配置"""
    print(f"\n🧪 测试: {name}")
    print(f"参数: {' '.join(args)}")
    
    start_time = time.time()
    try:
        result = subprocess.run([
            "python", "dist_main_with_feedback.py", "--epoch=5"
        ] + args, capture_output=True, text=True, timeout=timeout)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            stats = extract_training_stats(result.stdout)
            print(f"✅ 成功 - 耗时: {duration:.1f}s")
            print(f"   最终收敛节点: {stats['final_converged']}")
            if stats['converged_nodes']:
                print(f"   收敛进展: {' → '.join(map(str, stats['converged_nodes']))}")
            return True, stats, duration
        else:
            print(f"❌ 失败: {result.stderr[:100]}")
            return False, {}, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 超时 ({timeout}s)")
        return False, {}, timeout
    except Exception as e:
        print(f"💥 错误: {e}")
        return False, {}, 0

def main():
    print("🎯 智能采样效果测试")
    print("="*50)
    
    # 测试配置
    tests = [
        ("基准测试 (无反馈)", ["--disable_feedback"]),
        ("随机采样 (旧方式)", ["--enable_feedback", "--sampling_strategy=no_importance", "--similarity_threshold=0.8"]),
        ("智能采样 (新方式)", ["--enable_feedback", "--sampling_strategy=convergence_aware", "--similarity_threshold=0.70"])
    ]
    
    results = []
    
    for name, args in tests:
        success, stats, duration = run_test_config(name, args)
        results.append({
            'name': name,
            'success': success,
            'stats': stats,
            'duration': duration
        })
        
        # 间隔一下避免系统过载
        time.sleep(5)
    
    # 结果对比
    print(f"\n📊 结果对比")
    print("="*70)
    print(f"{'配置':<25} {'时间(s)':<8} {'收敛节点':<10} {'状态'}")
    print("-"*70)
    
    baseline_time = None
    for result in results:
        status = "✅" if result['success'] else "❌"
        converged = result['stats'].get('final_converged', 0) if result['success'] else 0
        
        print(f"{result['name']:<25} {result['duration']:<8.1f} {converged:<10} {status}")
        
        if '基准测试' in result['name'] and result['success']:
            baseline_time = result['duration']
    
    # 分析时间节约
    if baseline_time:
        print(f"\n⚡ 性能分析 (vs 基准: {baseline_time:.1f}s)")
        print("-"*50)
        for result in results:
            if result['success'] and '基准测试' not in result['name']:
                savings = baseline_time - result['duration']
                savings_pct = (savings / baseline_time) * 100
                converged = result['stats'].get('final_converged', 0)
                
                print(f"{result['name']:<25}: {savings:+6.1f}s ({savings_pct:+5.1f}%) | {converged} nodes converged")

if __name__ == "__main__":
    main()