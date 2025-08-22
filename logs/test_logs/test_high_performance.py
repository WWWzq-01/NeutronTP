#!/usr/bin/env python3
"""
高性能版本测试脚本
"""

import torch
import time
from info_feedback_system import InfoFeedbackSystem

def test_performance_comparison():
    """测试性能对比"""
    print("=== 测试性能对比 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    num_nodes = 10000
    embedding_dim = 64
    num_epochs = 30
    batch_size = 2000
    
    # 创建测试数据
    node_ids = torch.arange(num_nodes, device=device)
    losses = torch.rand(num_nodes, device=device)
    gradients = torch.rand(num_nodes, device=device)
    embeddings = torch.randn(num_nodes, embedding_dim, device=device)
    
    # 测试1：简单收敛模式（高性能）
    print("\n--- 测试1：简单收敛模式（高性能） ---")
    simple_system = InfoFeedbackSystem(
        num_nodes=num_nodes,
        device=device,
        enable_feedback=True,
        feedback_batch_cap=1000,
        sampler_candidate_pool_size=3000,
        keep_sampling_stats=False,
        similarity_threshold=0.95,
        patience=3,
        min_epochs=2,
        sampling_strategy='no_importance',
        use_simple_convergence=True
    )
    
    start_time = time.time()
    for epoch in range(num_epochs):
        indices = torch.randperm(num_nodes, device=device)[:batch_size]
        batch_node_ids = node_ids[indices]
        batch_losses = losses[indices]
        batch_gradients = gradients[indices]
        batch_embeddings = embeddings[indices]
        
        feedback_info = simple_system.process_feedback(
            batch_node_ids, batch_losses, batch_gradients, batch_embeddings, epoch
        )
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: {feedback_info['converged_nodes_count']} nodes converged, "
                  f"feedback_batch_size: {feedback_info.get('feedback_batch_size', 'N/A')}")
    
    simple_time = time.time() - start_time
    print(f"  简单收敛模式总时间: {simple_time:.2f}s")
    
    # 测试2：完整EMA模式（原始）
    print("\n--- 测试2：完整EMA模式（原始） ---")
    ema_system = InfoFeedbackSystem(
        num_nodes=num_nodes,
        device=device,
        enable_feedback=True,
        feedback_batch_cap=1000,
        sampler_candidate_pool_size=3000,
        keep_sampling_stats=False,
        similarity_threshold=0.95,
        patience=3,
        min_epochs=2,
        sampling_strategy='no_importance',
        use_simple_convergence=False
    )
    
    start_time = time.time()
    for epoch in range(num_epochs):
        indices = torch.randperm(num_nodes, device=device)[:batch_size]
        batch_node_ids = node_ids[indices]
        batch_losses = losses[indices]
        batch_gradients = gradients[indices]
        batch_embeddings = embeddings[indices]
        
        feedback_info = ema_system.process_feedback(
            batch_node_ids, batch_losses, batch_gradients, batch_embeddings, epoch
        )
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: {feedback_info['converged_nodes_count']} nodes converged, "
                  f"feedback_batch_size: {feedback_info.get('feedback_batch_size', 'N/A')}")
    
    ema_time = time.time() - start_time
    print(f"  完整EMA模式总时间: {ema_time:.2f}s")
    
    # 性能对比
    print(f"\n=== 性能对比结果 ===")
    print(f"简单收敛模式: {simple_time:.2f}s")
    print(f"完整EMA模式: {ema_time:.2f}s")
    
    if simple_time < ema_time:
        improvement = (ema_time - simple_time) / ema_time * 100
        print(f"✅ 简单收敛模式性能提升: {improvement:.1f}%")
    else:
        degradation = (simple_time - ema_time) / ema_time * 100
        print(f"❌ 简单收敛模式性能下降: {degradation:.1f}%")
    
    # 获取最终状态对比
    simple_status = simple_system.get_status()
    ema_status = ema_system.get_status()
    
    print(f"\n=== 收敛效果对比 ===")
    print(f"简单收敛模式: {simple_status['converged_nodes_count']} nodes converged")
    print(f"完整EMA模式: {ema_status['converged_nodes_count']} nodes converged")
    
    return simple_time, ema_time

def test_feedback_batch_cap_impact():
    """测试反馈批大小上限的影响（高性能版本）"""
    print("\n=== 测试反馈批大小上限的影响（高性能版本） ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = 15000
    embedding_dim = 64
    num_epochs = 20
    batch_size = 3000
    
    # 创建测试数据
    node_ids = torch.arange(num_nodes, device=device)
    losses = torch.rand(num_nodes, device=device)
    gradients = torch.rand(num_nodes, device=device)
    embeddings = torch.randn(num_nodes, embedding_dim, device=device)
    
    # 测试不同的反馈批大小上限
    batch_caps = [None, 5000, 2000, 1000]
    
    for batch_cap in batch_caps:
        print(f"\n--- 测试反馈批大小上限: {batch_cap} ---")
        
        system = InfoFeedbackSystem(
            num_nodes=num_nodes,
            device=device,
            enable_feedback=True,
            feedback_batch_cap=batch_cap,
            sampler_candidate_pool_size=5000,
            keep_sampling_stats=False,
            similarity_threshold=0.95,
            patience=3,
            min_epochs=2,
            sampling_strategy='no_importance',
            use_simple_convergence=True  # 使用高性能模式
        )
        
        start_time = time.time()
        for epoch in range(num_epochs):
            indices = torch.randperm(num_nodes, device=device)[:batch_size]
            batch_node_ids = node_ids[indices]
            batch_losses = losses[indices]
            batch_gradients = gradients[indices]
            batch_embeddings = embeddings[indices]
            
            feedback_info = system.process_feedback(
                batch_node_ids, batch_losses, batch_gradients, batch_embeddings, epoch
            )
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: feedback_batch_size={feedback_info['feedback_batch_size']}, "
                      f"capped={feedback_info['feedback_batch_capped']}")
        
        total_time = time.time() - start_time
        print(f"  总时间: {total_time:.2f}s")
        
        final_status = system.get_status()
        print(f"  收敛节点数: {final_status['converged_nodes_count']}")

def main():
    """主函数"""
    print("开始测试高性能版本...")
    
    # 测试性能对比
    simple_time, ema_time = test_performance_comparison()
    
    # 测试反馈批大小上限影响
    test_feedback_batch_cap_impact()
    
    print(f"\n=== 测试完成 ===")
    print(f"简单收敛模式: {simple_time:.2f}s")
    print(f"完整EMA模式: {ema_time:.2f}s")

if __name__ == "__main__":
    main()
