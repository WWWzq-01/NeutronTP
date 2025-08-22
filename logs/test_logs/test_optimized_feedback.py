#!/usr/bin/env python3
"""
测试优化后的信息互馈系统性能
"""

import torch
import time
import numpy as np
from info_feedback_system import InfoFeedbackSystem
from feedback_config import FeedbackConfigTemplates

def test_ema_vs_deque_performance():
    """测试EMA vs deque的性能差异"""
    print("=== 测试EMA vs deque性能差异 ===")
    
    num_nodes = 10000
    embedding_dim = 64
    num_epochs = 50
    batch_size = 1000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    node_ids = torch.arange(num_nodes, device=device)
    losses = torch.rand(num_nodes, device=device)
    gradients = torch.rand(num_nodes, device=device)
    embeddings = torch.randn(num_nodes, embedding_dim, device=device)
    
    # 测试EMA版本（优化后）
    print("\n--- 测试EMA版本（优化后） ---")
    ema_system = InfoFeedbackSystem(
        num_nodes=num_nodes,
        device=device,
        enable_feedback=True,
        feedback_batch_cap=5000,  # 限制反馈批大小
        sampler_candidate_pool_size=5000,  # 限制候选池
        keep_sampling_stats=False,  # 关闭统计
        similarity_threshold=0.95,
        patience=5,
        min_epochs=3,
        sampling_strategy='no_importance'
    )
    
    start_time = time.time()
    for epoch in range(num_epochs):
        # 随机选择batch
        indices = torch.randperm(num_nodes, device=device)[:batch_size]
        batch_node_ids = node_ids[indices]
        batch_losses = losses[indices]
        batch_gradients = gradients[indices]
        batch_embeddings = embeddings[indices]
        
        # 处理反馈
        feedback_info = ema_system.process_feedback(
            batch_node_ids, batch_losses, batch_gradients, batch_embeddings, epoch
        )
        
        # 获取自适应batch
        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        adaptive_batch = ema_system.get_adaptive_batch(batch_size, epoch, train_mask)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {feedback_info['converged_nodes_count']} nodes converged, "
                  f"feedback_batch_size: {feedback_info.get('feedback_batch_size', 'N/A')}")
    
    ema_time = time.time() - start_time
    print(f"EMA版本总时间: {ema_time:.2f}s")
    
    # 测试采样性能
    print("\n--- 测试采样性能 ---")
    start_time = time.time()
    for _ in range(100):
        train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        adaptive_batch = ema_system.get_adaptive_batch(batch_size, 0, train_mask)
    sampling_time = time.time() - start_time
    print(f"100次采样时间: {sampling_time:.4f}s")
    
    # 获取最终状态
    final_status = ema_system.get_status()
    print(f"\n最终状态: {final_status}")
    
    return ema_time, sampling_time

def test_feedback_batch_cap_impact():
    """测试反馈批大小上限的影响"""
    print("\n=== 测试反馈批大小上限的影响 ===")
    
    num_nodes = 20000
    embedding_dim = 64
    num_epochs = 20
    batch_size = 5000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    node_ids = torch.arange(num_nodes, device=device)
    losses = torch.rand(num_nodes, device=device)
    gradients = torch.rand(num_nodes, device=device)
    embeddings = torch.randn(num_nodes, embedding_dim, device=device)
    
    # 测试不同的反馈批大小上限
    batch_caps = [None, 10000, 5000, 2000]
    
    for batch_cap in batch_caps:
        print(f"\n--- 测试反馈批大小上限: {batch_cap} ---")
        
        system = InfoFeedbackSystem(
            num_nodes=num_nodes,
            device=device,
            enable_feedback=True,
            feedback_batch_cap=batch_cap,
            sampler_candidate_pool_size=10000,
            keep_sampling_stats=False,
            similarity_threshold=0.95,
            patience=5,
            min_epochs=3,
            sampling_strategy='no_importance'
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

def test_config_templates():
    """测试配置模板的性能差异"""
    print("\n=== 测试配置模板的性能差异 ===")
    
    num_nodes = 15000
    embedding_dim = 64
    num_epochs = 30
    batch_size = 2000
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    node_ids = torch.arange(num_nodes, device=device)
    losses = torch.rand(num_nodes, device=device)
    gradients = torch.rand(num_nodes, device=device)
    embeddings = torch.randn(num_nodes, embedding_dim, device=device)
    
    # 测试不同的配置模板
    templates = [
        ('conservative', FeedbackConfigTemplates.conservative()),
        ('aggressive', FeedbackConfigTemplates.aggressive()),
        ('performance_optimized', FeedbackConfigTemplates.performance_optimized()),
        ('balanced', FeedbackConfigTemplates.balanced())
    ]
    
    for template_name, config in templates:
        print(f"\n--- 测试配置模板: {template_name} ---")
        
        system = InfoFeedbackSystem(
            num_nodes=num_nodes,
            device=device,
            enable_feedback=True,
            feedback_batch_cap=config['performance']['feedback_batch_cap'],
            sampler_candidate_pool_size=config['sampling']['candidate_pool_size'],
            keep_sampling_stats=config['sampling']['keep_sampling_stats'],
            similarity_threshold=config['convergence']['similarity_threshold'],
            patience=config['convergence']['patience'],
            min_epochs=config['convergence']['min_epochs'],
            sampling_strategy='no_importance'
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
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: {feedback_info['converged_nodes_count']} nodes converged")
        
        total_time = time.time() - start_time
        print(f"  总时间: {total_time:.2f}s")
        
        final_status = system.get_status()
        print(f"  收敛节点数: {final_status['converged_nodes_count']}")
        print(f"  反馈批大小上限: {config['performance']['feedback_batch_cap']}")
        print(f"  候选池大小: {config['sampling']['candidate_pool_size']}")

def main():
    """主测试函数"""
    print("开始测试优化后的信息互馈系统性能...")
    
    # 测试EMA vs deque性能
    ema_time, sampling_time = test_ema_vs_deque_performance()
    
    # 测试反馈批大小上限影响
    test_feedback_batch_cap_impact()
    
    # 测试配置模板
    test_config_templates()
    
    print("\n=== 测试完成 ===")
    print(f"EMA版本总训练时间: {ema_time:.2f}s")
    print(f"采样性能: {sampling_time:.4f}s for 100 samples")

if __name__ == "__main__":
    main()
