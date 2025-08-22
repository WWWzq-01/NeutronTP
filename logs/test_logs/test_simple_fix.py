#!/usr/bin/env python3
"""
简单测试脚本，验证设备问题的修复
"""

import torch
from info_feedback_system import InfoFeedbackSystem

def test_device_consistency():
    """测试设备一致性"""
    print("=== 测试设备一致性 ===")
    
    # 测试CUDA设备
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"使用CUDA设备: {device}")
    else:
        device = torch.device('cpu')
        print(f"使用CPU设备: {device}")
    
    num_nodes = 1000
    embedding_dim = 32
    
    # 创建系统
    system = InfoFeedbackSystem(
        num_nodes=num_nodes,
        device=device,
        enable_feedback=True,
        feedback_batch_cap=500,
        sampler_candidate_pool_size=1000,
        keep_sampling_stats=False,
        similarity_threshold=0.95,
        patience=3,
        min_epochs=2,
        sampling_strategy='no_importance'
    )
    
    print("系统创建成功")
    
    # 创建测试数据
    batch_size = 100
    node_ids = torch.arange(batch_size, device=device)
    losses = torch.rand(batch_size, device=device)
    gradients = torch.rand(batch_size, device=device)
    embeddings = torch.randn(batch_size, embedding_dim, device=device)
    
    print(f"测试数据创建成功，设备: {node_ids.device}")
    
    # 测试反馈处理
    try:
        feedback_info = system.process_feedback(
            node_ids, losses, gradients, embeddings, 0
        )
        print(f"反馈处理成功: {feedback_info}")
    except Exception as e:
        print(f"反馈处理失败: {e}")
        return False
    
    # 测试采样
    try:
        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        batch = system.get_adaptive_batch(batch_size, 0, train_mask)
        print(f"采样成功，batch大小: {len(batch)}")
    except Exception as e:
        print(f"采样失败: {e}")
        return False
    
    # 测试状态获取
    try:
        status = system.get_status()
        print(f"状态获取成功: {status}")
    except Exception as e:
        print(f"状态获取失败: {e}")
        return False
    
    print("所有测试通过！")
    return True

def main():
    """主函数"""
    print("开始测试设备一致性...")
    
    success = test_device_consistency()
    
    if success:
        print("\n✅ 测试成功！设备问题已修复")
    else:
        print("\n❌ 测试失败，仍有问题需要修复")

if __name__ == "__main__":
    main()
