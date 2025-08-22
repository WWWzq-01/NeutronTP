#!/usr/bin/env python3
"""
测试信息互馈系统的修复版本
"""

import torch
import torch.nn as nn
import numpy as np
from info_feedback_system import InfoFeedbackSystem, NodeConvergenceTracker
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_convergence_tracker():
    """测试节点收敛跟踪器"""
    print("测试节点收敛跟踪器...")
    
    # 创建跟踪器
    tracker = NodeConvergenceTracker(num_nodes=100, embedding_dim=32)
    
    # 模拟数据
    node_ids = torch.tensor([0, 1, 2, 3, 4])
    losses = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])
    gradients = torch.tensor([0.1, 0.05, 0.15, 0.02, 0.12])
    embeddings = torch.randn(5, 32)
    
    # 测试更新
    try:
        tracker.update_node_info(node_ids, losses, gradients, embeddings, epoch=1)
        print("✓ 节点信息更新成功")
        
        # 测试多次更新
        for epoch in range(2, 7):
            # 模拟损失逐渐减小
            losses = losses * 0.9
            gradients = gradients * 0.9
            embeddings = embeddings * 0.9
            tracker.update_node_info(node_ids, losses, gradients, embeddings, epoch)
        
        print(f"✓ 多次更新成功，当前收敛节点数: {len(tracker.get_converged_nodes())}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False
    
    return True

def test_feedback_system():
    """测试信息互馈系统"""
    print("\n测试信息互馈系统...")
    
    # 创建系统
    device = torch.device('cpu')
    feedback_system = InfoFeedbackSystem(num_nodes=200, device=device, enable_feedback=True)
    
    # 模拟训练数据
    train_mask = torch.zeros(200, dtype=torch.bool)
    train_mask[:150] = True  # 前150个节点为训练节点
    
    try:
        # 测试获取自适应batch
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=50, epoch=1, train_mask=train_mask)
        print(f"✓ 获取自适应batch成功，大小: {len(batch_nodes)}")
        
        # 测试处理反馈
        batch_losses = torch.rand(len(batch_nodes))
        batch_gradients = torch.rand(len(batch_nodes))
        batch_embeddings = torch.randn(len(batch_nodes), 64)
        
        feedback_info = feedback_system.process_feedback(
            batch_nodes, batch_losses, batch_gradients, batch_embeddings, epoch=1
        )
        
        if feedback_info:
            print(f"✓ 处理反馈成功，收敛率: {feedback_info['convergence_rate']:.3f}")
        else:
            print("✗ 反馈处理失败")
            return False
        
        # 测试系统状态
        status = feedback_system.get_status()
        print(f"✓ 获取系统状态成功: {status}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_edge_cases():
    """测试边界情况"""
    print("\n测试边界情况...")
    
    device = torch.device('cpu')
    feedback_system = InfoFeedbackSystem(num_nodes=100, device=device, enable_feedback=True)
    
    # 测试空数据
    try:
        empty_mask = torch.zeros(100, dtype=torch.bool)
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=10, epoch=1, train_mask=empty_mask)
        print(f"✓ 空训练掩码处理成功，batch大小: {len(batch_nodes)}")
        
        # 测试None值
        feedback_info = feedback_system.process_feedback(
            torch.tensor([]), torch.tensor([]), None, None, epoch=1
        )
        print("✓ None值处理成功")
        
    except Exception as e:
        print(f"✗ 边界情况测试失败: {e}")
        return False
    
    return True

def test_performance():
    """测试性能"""
    print("\n测试性能...")
    
    import time
    
    device = torch.device('cpu')
    feedback_system = InfoFeedbackSystem(num_nodes=10000, device=device, enable_feedback=True)
    
    # 创建大型训练掩码
    train_mask = torch.zeros(10000, dtype=torch.bool)
    train_mask[:8000] = True
    
    # 测试采样性能
    start_time = time.time()
    for epoch in range(10):
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=1000, epoch=epoch, train_mask=train_mask)
        
        # 模拟反馈
        batch_losses = torch.rand(len(batch_nodes))
        batch_gradients = torch.rand(len(batch_nodes))
        batch_embeddings = torch.randn(len(batch_nodes), 128)
        
        feedback_system.process_feedback(
            batch_nodes, batch_losses, batch_gradients, batch_embeddings, epoch
        )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"✓ 性能测试完成，10个epoch耗时: {elapsed_time:.3f}秒")
    print(f"  平均每个epoch: {elapsed_time/10:.3f}秒")
    
    return True

def main():
    """主测试函数"""
    print("开始测试信息互馈系统...")
    print("=" * 50)
    
    tests = [
        test_convergence_tracker,
        test_feedback_system,
        test_edge_cases,
        test_performance
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
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统修复成功。")
    else:
        print("❌ 部分测试失败，需要进一步修复。")
    
    return passed == total

if __name__ == "__main__":
    main()
