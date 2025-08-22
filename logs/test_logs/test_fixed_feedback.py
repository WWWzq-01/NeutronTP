#!/usr/bin/env python3
"""
测试修复后的信息互馈系统
"""

import torch
import numpy as np
from info_feedback_system import InfoFeedbackSystem, NodeConvergenceTracker, AdaptiveSampler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_edge_cases_fixed():
    """测试修复后的边界情况"""
    print("测试修复后的边界情况...")
    
    device = torch.device('cpu')
    feedback_system = InfoFeedbackSystem(num_nodes=100, device=device, enable_feedback=True)
    
    try:
        # 测试空训练掩码
        print("  测试空训练掩码...")
        empty_mask = torch.zeros(100, dtype=torch.bool)
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=10, epoch=1, train_mask=empty_mask)
        
        if len(batch_nodes) == 0:
            print("    ✓ 空训练掩码处理成功，返回空batch")
        else:
            print(f"    ✗ 空训练掩码处理失败，返回了 {len(batch_nodes)} 个节点")
            return False
        
        # 测试None值处理
        print("  测试None值处理...")
        feedback_info = feedback_system.process_feedback(
            torch.tensor([]), torch.tensor([]), None, None, epoch=1
        )
        print("    ✓ None值处理成功")
        
        # 测试小batch_size
        print("  测试小batch_size...")
        small_mask = torch.zeros(100, dtype=torch.bool)
        small_mask[:5] = True  # 只有5个训练节点
        
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=10, epoch=1, train_mask=small_mask)
        if len(batch_nodes) == 5:  # 应该返回所有5个节点
            print("    ✓ 小batch_size处理成功")
        else:
            print(f"    ✗ 小batch_size处理失败，期望5个节点，实际{len(batch_nodes)}个")
            return False
        
        print("✓ 边界情况测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 边界情况测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampling_strategies_fixed():
    """测试修复后的采样策略"""
    print("\n测试修复后的采样策略...")
    
    num_nodes = 500
    device = torch.device('cpu')
    
    # 创建收敛跟踪器
    convergence_tracker = NodeConvergenceTracker(num_nodes, embedding_dim=64)
    
    # 测试不同的采样策略
    strategies = ['adaptive_importance', 'convergence_aware', 'random']
    
    for strategy in strategies:
        print(f"  测试策略: {strategy}")
        
        try:
            # 创建采样器
            sampler = AdaptiveSampler(num_nodes, convergence_tracker, strategy)
            
            # 模拟一些节点收敛
            for i in range(100):
                convergence_tracker.converged_nodes.add(i)
            
            # 创建训练掩码
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[:200] = True  # 200个训练节点
            
            # 测试采样
            batch_size = 100
            epoch = 10
            
            sampled_nodes = sampler.sample_nodes(batch_size, epoch, train_mask)
            
            if len(sampled_nodes) == batch_size:
                converged_in_batch = sum(1 for n in sampled_nodes if n.item() in convergence_tracker.converged_nodes)
                print(f"    ✓ 采样成功，batch大小: {len(sampled_nodes)}, 收敛节点: {converged_in_batch}")
            else:
                print(f"    ✗ 采样失败，期望{batch_size}个节点，实际{len(sampled_nodes)}个")
                return False
                
        except Exception as e:
            print(f"    ✗ 策略 {strategy} 测试失败: {e}")
            return False
    
    print("✓ 采样策略测试通过")
    return True

def test_feedback_processing_fixed():
    """测试修复后的反馈处理"""
    print("\n测试修复后的反馈处理...")
    
    device = torch.device('cpu')
    feedback_system = InfoFeedbackSystem(num_nodes=200, device=device, enable_feedback=True)
    
    try:
        # 创建训练掩码
        train_mask = torch.zeros(200, dtype=torch.bool)
        train_mask[:150] = True  # 150个训练节点
        
        # 测试正常情况
        print("  测试正常反馈处理...")
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=50, epoch=1, train_mask=train_mask)
        
        if len(batch_nodes) > 0:
            batch_losses = torch.rand(len(batch_nodes))
            batch_gradients = batch_losses.clone()  # 使用损失作为梯度
            batch_embeddings = torch.randn(len(batch_nodes), 64)
            
            feedback_info = feedback_system.process_feedback(
                batch_nodes, batch_losses, batch_gradients, batch_embeddings, epoch=1
            )
            
            if feedback_info:
                print(f"    ✓ 正常反馈处理成功，收敛率: {feedback_info['convergence_rate']:.3f}")
            else:
                print("    ✗ 正常反馈处理失败")
                return False
        else:
            print("    ✗ 无法获取batch")
            return False
        
        # 测试空batch
        print("  测试空batch处理...")
        empty_batch = torch.tensor([], dtype=torch.long)
        empty_losses = torch.tensor([])
        empty_gradients = torch.tensor([])
        empty_embeddings = torch.tensor([])
        
        feedback_info = feedback_system.process_feedback(
            empty_batch, empty_losses, empty_gradients, empty_embeddings, epoch=1
        )
        
        if feedback_info:
            print(f"    ✓ 空batch处理成功，收敛率: {feedback_info['convergence_rate']:.3f}")
        else:
            print("    ✗ 空batch处理失败")
            return False
        
        print("✓ 反馈处理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 反馈处理测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修复后的信息互馈系统...")
    print("=" * 60)
    
    tests = [
        test_edge_cases_fixed,
        test_sampling_strategies_fixed,
        test_feedback_processing_fixed
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"测试 {test_func.__name__} 失败")
        except Exception as e:
            print(f"测试 {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！边界情况修复成功。")
        print("\n现在可以安全地运行分布式训练了。")
    else:
        print("❌ 部分测试失败，需要进一步修复。")
    
    return passed == total

if __name__ == "__main__":
    main()
