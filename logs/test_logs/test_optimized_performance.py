#!/usr/bin/env python3
"""
测试优化后信息互馈系统的性能
"""
import torch
import time
from info_feedback_system import InfoFeedbackSystem, compute_cosine_similarity_batch

def test_batch_similarity():
    """测试批量相似性计算性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 生成测试数据
    batch_size = 1000
    embed_dim = 128
    embed1 = torch.randn(batch_size, embed_dim, device=device)
    embed2 = torch.randn(batch_size, embed_dim, device=device)
    
    # 预热
    for _ in range(10):
        _ = compute_cosine_similarity_batch(embed1, embed2)
    
    # 性能测试
    start_time = time.time()
    iterations = 100
    
    for _ in range(iterations):
        similarities = compute_cosine_similarity_batch(embed1, embed2)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations * 1000  # ms
    print(f"批量相似性计算平均耗时: {avg_time:.3f}ms (batch_size={batch_size}, embed_dim={embed_dim})")
    return avg_time

def test_feedback_system_performance():
    """测试信息互馈系统整体性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建系统
    num_nodes = 10000
    system = InfoFeedbackSystem(
        num_nodes=num_nodes,
        device=device,
        feedback_batch_cap=1000,  # 限制批量大小
        use_simple_convergence=True  # 使用简单收敛判断
    )
    
    # 生成测试数据
    batch_size = 1000
    embed_dim = 64
    
    node_ids = torch.randint(0, num_nodes, (batch_size,), device=device)
    losses = torch.randn(batch_size, device=device)
    gradients = torch.randn(batch_size, device=device) 
    embeddings = torch.randn(batch_size, embed_dim, device=device)
    
    # 预热
    for epoch in range(5):
        system.process_feedback(node_ids, losses, gradients, embeddings, epoch)
    
    # 性能测试
    start_time = time.time()
    test_epochs = 20
    
    for epoch in range(test_epochs):
        feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, epoch)
        
        # 模拟自适应采样
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[:num_nodes//2] = True
        batch_nodes = system.get_adaptive_batch(batch_size, epoch, train_mask)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / test_epochs * 1000  # ms
    print(f"信息互馈系统平均处理时间: {avg_time:.3f}ms/epoch (num_nodes={num_nodes}, batch_size={batch_size})")
    
    # 打印系统状态
    status = system.get_status()
    print(f"系统状态: {status}")
    
    return avg_time

if __name__ == "__main__":
    print("=== 优化后信息互馈系统性能测试 ===")
    print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()
    
    # 测试批量相似性计算
    similarity_time = test_batch_similarity()
    print()
    
    # 测试完整系统性能
    system_time = test_feedback_system_performance()
    print()
    
    print("=== 测试完成 ===")
    print(f"批量相似性计算: {similarity_time:.3f}ms")
    print(f"系统整体处理: {system_time:.3f}ms/epoch")
    
    # 性能评估
    if system_time < 50:  # 小于50ms认为性能良好
        print("✅ 性能优化成功！系统开销控制在合理范围内")
    elif system_time < 100:
        print("⚠️  性能可接受，但仍有优化空间")
    else:
        print("❌ 性能需要进一步优化")