#!/usr/bin/env python3
"""
基本功能测试脚本
"""

import torch
import time
from info_feedback_system import InfoFeedbackSystem

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试基本功能 ===")
    
    # 使用CPU设备进行测试
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    num_nodes = 1000
    embedding_dim = 32
    
    try:
        # 创建系统
        system = InfoFeedbackSystem(
            num_nodes=num_nodes,
            device=device,
            enable_feedback=True,
            feedback_batch_cap=200,
            sampler_candidate_pool_size=500,
            keep_sampling_stats=False,
            similarity_threshold=0.95,
            patience=3,
            min_epochs=2,
            sampling_strategy='no_importance'
        )
        print("✅ 系统创建成功")
        
        # 创建测试数据
        batch_size = 100
        node_ids = torch.arange(batch_size, device=device)
        losses = torch.rand(batch_size, device=device)
        gradients = torch.rand(batch_size, device=device)
        embeddings = torch.randn(batch_size, embedding_dim, device=device)
        print("✅ 测试数据创建成功")
        
        # 测试反馈处理
        print("\n--- 测试反馈处理 ---")
        for epoch in range(5):
            feedback_info = system.process_feedback(
                node_ids, losses, gradients, embeddings, epoch
            )
            print(f"Epoch {epoch}: {feedback_info['converged_nodes_count']} nodes converged, "
                  f"feedback_batch_size: {feedback_info.get('feedback_batch_size', 'N/A')}")
        
        # 测试采样
        print("\n--- 测试采样 ---")
        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        batch = system.get_adaptive_batch(batch_size, 5, train_mask)
        print(f"采样成功，batch大小: {len(batch)}")
        
        # 测试状态获取
        print("\n--- 测试状态获取 ---")
        status = system.get_status()
        print(f"状态: {status}")
        
        print("\n✅ 所有基本功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cuda_functionality():
    """测试CUDA功能（如果可用）"""
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过CUDA测试")
        return True
    
    print("\n=== 测试CUDA功能 ===")
    
    device = torch.device('cuda:0')
    print(f"使用CUDA设备: {device}")
    
    num_nodes = 500
    embedding_dim = 16
    
    try:
        # 创建系统
        system = InfoFeedbackSystem(
            num_nodes=num_nodes,
            device=device,
            enable_feedback=True,
            feedback_batch_cap=100,
            sampler_candidate_pool_size=200,
            keep_sampling_stats=False,
            similarity_threshold=0.95,
            patience=3,
            min_epochs=2,
            sampling_strategy='no_importance'
        )
        print("✅ CUDA系统创建成功")
        
        # 创建测试数据
        batch_size = 50
        node_ids = torch.arange(batch_size, device=device)
        losses = torch.rand(batch_size, device=device)
        gradients = torch.rand(batch_size, device=device)
        embeddings = torch.randn(batch_size, embedding_dim, device=device)
        print("✅ CUDA测试数据创建成功")
        
        # 测试反馈处理
        feedback_info = system.process_feedback(
            node_ids, losses, gradients, embeddings, 0
        )
        print(f"✅ CUDA反馈处理成功: {feedback_info['converged_nodes_count']} nodes converged")
        
        # 测试采样
        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        batch = system.get_adaptive_batch(batch_size, 0, train_mask)
        print(f"✅ CUDA采样成功，batch大小: {len(batch)}")
        
        print("✅ 所有CUDA功能测试通过！")
        return True
        
    except Exception as e:
        print(f"\n❌ CUDA测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始测试信息互馈系统...")
    
    # 测试基本功能
    basic_success = test_basic_functionality()
    
    # 测试CUDA功能
    cuda_success = test_cuda_functionality()
    
    if basic_success and cuda_success:
        print("\n🎉 所有测试通过！系统工作正常")
    else:
        print("\n💥 部分测试失败，需要进一步修复")

if __name__ == "__main__":
    main()
