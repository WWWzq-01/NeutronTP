#!/usr/bin/env python3
"""
全面的信息互馈系统测试脚本
"""

import torch
import numpy as np
from info_feedback_system import InfoFeedbackSystem, NodeConvergenceTracker, AdaptiveSampler
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name_s)s - %(levelname)s - %(message)s')

def test_empty_tensor_handling():
    """测试空张量处理"""
    print("测试空张量处理...")
    
    try:
        # 创建收敛跟踪器
        tracker = NodeConvergenceTracker(num_nodes=100, embedding_dim=64)
        
        # 测试空张量
        empty_node_ids = torch.tensor([], dtype=torch.long)
        empty_losses = torch.tensor([])
        empty_gradients = torch.tensor([])
        empty_embeddings = torch.tensor([])
        
        # 这应该不会抛出异常
        tracker.update_node_info(empty_node_ids, empty_losses, empty_gradients, empty_embeddings, epoch=1)
        print("  ✓ 空张量处理成功")
        
        # 测试None值
        tracker.update_node_info(empty_node_ids, empty_losses, None, None, epoch=1)
        print("  ✓ None值处理成功")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 空张量处理失败: {e}")
        return False

def test_edge_case_sampling():
    """测试边界情况采样"""
    print("\n测试边界情况采样...")
    
    try:
        num_nodes = 100
        device = torch.device('cpu')
        
        # 创建信息互馈系统
        feedback_system = InfoFeedbackSystem(num_nodes, device, enable_feedback=True)
        
        # 测试1: 空训练掩码
        print("  测试1: 空训练掩码")
        empty_mask = torch.zeros(num_nodes, dtype=torch.bool)
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=10, epoch=1, train_mask=empty_mask)
        
        if len(batch_nodes) == 0:
            print("    ✓ 空训练掩码处理成功")
        else:
            print(f"    ✗ 空训练掩码处理失败，返回了 {len(batch_nodes)} 个节点")
            return False
        
        # 测试2: 小训练掩码
        print("  测试2: 小训练掩码")
        small_mask = torch.zeros(num_nodes, dtype=torch.bool)
        small_mask[:5] = True  # 只有5个训练节点
        
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=10, epoch=1, train_mask=small_mask)
        if len(batch_nodes) == 5:
            print("    ✓ 小训练掩码处理成功")
        else:
            print(f"    ✗ 小训练掩码处理失败，期望5个节点，实际{len(batch_nodes)}个")
            return False
        
        # 测试3: 正常训练掩码
        print("  测试3: 正常训练掩码")
        normal_mask = torch.zeros(num_nodes, dtype=torch.bool)
        normal_mask[:50] = True  # 50个训练节点
        
        batch_nodes = feedback_system.get_adaptive_batch(batch_size=20, epoch=1, train_mask=normal_mask)
        if len(batch_nodes) == 20:
            print("    ✓ 正常训练掩码处理成功")
        else:
            print(f"    ✗ 正常训练掩码处理失败，期望20个节点，实际{len(batch_nodes)}个")
            return False
        
        print("  ✓ 边界情况采样测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 边界情况采样测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feedback_processing_edge_cases():
    """测试反馈处理的边界情况"""
    print("\n测试反馈处理的边界情况...")
    
    try:
        device = torch.device('cpu')
        feedback_system = InfoFeedbackSystem(num_nodes=200, device=device, enable_feedback=True)
        
        # 测试1: 空batch反馈
        print("  测试1: 空batch反馈")
        empty_batch = torch.tensor([], dtype=torch.long)
        empty_losses = torch.tensor([])
        empty_gradients = torch.tensor([])
        empty_embeddings = torch.tensor([])
        
        feedback_info = feedback_system.process_feedback(
            empty_batch, empty_losses, empty_gradients, empty_embeddings, epoch=1
        )
        
        if feedback_info and feedback_info['convergence_rate'] == 0.0:
            print("    ✓ 空batch反馈处理成功")
        else:
            print("    ✗ 空batch反馈处理失败")
            return False
        
        # 测试2: 单个节点反馈
        print("  测试2: 单个节点反馈")
        single_batch = torch.tensor([0], dtype=torch.long)
        single_losses = torch.tensor([0.5])
        single_gradients = torch.tensor([0.1])
        single_embeddings = torch.randn(1, 64)
        
        feedback_info = feedback_system.process_feedback(
            single_batch, single_losses, single_gradients, single_embeddings, epoch=1
        )
        
        if feedback_info:
            print(f"    ✓ 单个节点反馈处理成功，收敛率: {feedback_info['convergence_rate']:.3f}")
        else:
            print("    ✗ 单个节点反馈处理失败")
            return False
        
        # 测试3: 正常batch反馈
        print("  测试3: 正常batch反馈")
        normal_batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        normal_losses = torch.tensor([0.5, 0.3, 0.8, 0.2, 0.6])
        normal_gradients = torch.tensor([0.1, 0.05, 0.15, 0.02, 0.12])
        normal_embeddings = torch.randn(5, 64)
        
        feedback_info = feedback_system.process_feedback(
            normal_batch, normal_losses, normal_gradients, normal_embeddings, epoch=1
        )
        
        if feedback_info:
            print(f"    ✓ 正常batch反馈处理成功，收敛率: {feedback_info['convergence_rate']:.3f}")
        else:
            print("    ✗ 正常batch反馈处理失败")
            return False
        
        print("  ✓ 反馈处理边界情况测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 反馈处理边界情况测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sampling_strategies_comprehensive():
    """测试所有采样策略"""
    print("\n测试所有采样策略...")
    
    try:
        num_nodes = 500
        device = torch.device('cpu')
        
        # 创建收敛跟踪器
        convergence_tracker = NodeConvergenceTracker(num_nodes, embedding_dim=64)
        
        # 模拟一些节点收敛
        for i in range(100):
            convergence_tracker.converged_nodes.add(i)
        
        # 测试所有策略
        strategies = ['adaptive_importance', 'convergence_aware', 'random']
        
        for strategy in strategies:
            print(f"  测试策略: {strategy}")
            
            # 创建采样器
            sampler = AdaptiveSampler(num_nodes, convergence_tracker, strategy)
            
            # 创建训练掩码
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            train_mask[:200] = True  # 200个训练节点
            
            # 测试不同batch_size
            for batch_size in [10, 50, 100]:
                try:
                    sampled_nodes = sampler.sample_nodes(batch_size, epoch=10, train_mask=train_mask)
                    
                    if len(sampled_nodes) == batch_size:
                        converged_in_batch = sum(1 for n in sampled_nodes if n.item() in convergence_tracker.converged_nodes)
                        print(f"    ✓ batch_size={batch_size}: 采样成功，收敛节点: {converged_in_batch}")
                    else:
                        print(f"    ✗ batch_size={batch_size}: 采样失败，期望{batch_size}个节点，实际{len(sampled_nodes)}个")
                        return False
                        
                except Exception as e:
                    print(f"    ✗ batch_size={batch_size}: 采样异常: {e}")
                    return False
        
        print("  ✓ 采样策略测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 采样策略测试失败: {e}")
        return False

def test_system_integration():
    """测试系统集成"""
    print("\n测试系统集成...")
    
    try:
        device = torch.device('cpu')
        feedback_system = InfoFeedbackSystem(num_nodes=1000, device=device, enable_feedback=True)
        
        # 创建训练掩码
        train_mask = torch.zeros(1000, dtype=torch.bool)
        train_mask[:800] = True  # 800个训练节点
        
        # 模拟完整的训练流程
        print("  模拟完整训练流程...")
        
        for epoch in range(5):
            # 获取batch
            batch_nodes = feedback_system.get_adaptive_batch(batch_size=100, epoch=epoch, train_mask=train_mask)
            
            if len(batch_nodes) > 0:
                # 模拟训练反馈
                batch_losses = torch.rand(len(batch_nodes))
                batch_gradients = batch_losses.clone()
                batch_embeddings = torch.randn(len(batch_nodes), 64)
                
                # 处理反馈
                feedback_info = feedback_system.process_feedback(
                    batch_nodes, batch_losses, batch_gradients, batch_embeddings, epoch
                )
                
                if feedback_info:
                    print(f"    Epoch {epoch}: 收敛率 {feedback_info['convergence_rate']:.3f}")
                else:
                    print(f"    Epoch {epoch}: 反馈处理失败")
                    return False
            else:
                print(f"    Epoch {epoch}: 空batch")
        
        # 获取系统状态
        status = feedback_system.get_status()
        print(f"  最终系统状态: {status}")
        
        print("  ✓ 系统集成测试通过")
        return True
        
    except Exception as e:
        print(f"  ✗ 系统集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始全面测试信息互馈系统...")
    print("=" * 70)
    
    tests = [
        test_empty_tensor_handling,
        test_edge_case_sampling,
        test_feedback_processing_edge_cases,
        test_sampling_strategies_comprehensive,
        test_system_integration
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
    
    print("\n" + "=" * 70)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统修复完成，可以安全使用。")
        print("\n现在可以运行分布式训练了！")
    else:
        print("❌ 部分测试失败，需要进一步修复。")
    
    return passed == total

if __name__ == "__main__":
    main()
