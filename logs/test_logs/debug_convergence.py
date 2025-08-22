#!/usr/bin/env python3
"""
深度调试收敛检测问题
"""
import torch
from info_feedback_system import NodeConvergenceTracker

def debug_convergence_tracker():
    """调试收敛跟踪器的内部状态"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单的跟踪器
    tracker = NodeConvergenceTracker(
        num_nodes=100,
        patience=2,
        min_epochs=1,
        similarity_threshold=0.80,
        use_simple_convergence=True,
        device=device
    )
    
    print("=== 调试节点收敛跟踪器 ===")
    print(f"设备: {device}")
    print(f"参数: patience={tracker.patience}, min_epochs={tracker.min_epochs}")
    print(f"similarity_threshold={tracker.similarity_threshold}")
    print()
    
    # 测试数据
    batch_size = 10
    embed_dim = 16
    node_ids = torch.arange(batch_size, device=device)
    
    # 第一次更新
    print("=== 第一次更新 ===")
    embeddings1 = torch.randn(batch_size, embed_dim, device=device)
    losses1 = torch.randn(batch_size, device=device)
    gradients1 = torch.randn(batch_size, device=device)
    
    tracker.update_node_info(node_ids, losses1, gradients1, embeddings1, 0)
    
    print(f"更新后:")
    print(f"- update_count: {tracker.node_update_count[node_ids].tolist()}")
    print(f"- eligible_nodes数量: {len(tracker.eligible_nodes)}")
    print(f"- converged_nodes数量: {len(tracker.converged_nodes)}")
    print()
    
    # 第二次更新
    print("=== 第二次更新（稍微变化）===")
    embeddings2 = embeddings1 + torch.randn_like(embeddings1) * 0.1
    losses2 = torch.randn(batch_size, device=device)
    gradients2 = torch.randn(batch_size, device=device)
    
    tracker.update_node_info(node_ids, losses2, gradients2, embeddings2, 1)
    
    print(f"更新后:")
    print(f"- update_count: {tracker.node_update_count[node_ids].tolist()}")
    print(f"- eligible_nodes数量: {len(tracker.eligible_nodes)}")
    print(f"- converged_nodes数量: {len(tracker.converged_nodes)}")
    print()
    
    # 第三次更新（应该检查收敛）
    print("=== 第三次更新（相同嵌入，应该收敛）===")
    # 使用完全相同的嵌入
    embeddings3 = embeddings2.clone()
    losses3 = torch.randn(batch_size, device=device)
    gradients3 = torch.randn(batch_size, device=device)
    
    # 手动计算相似性验证
    if tracker.node_update_count[0] >= tracker.patience:
        current_embed = tracker.node_last_embeddings[0]
        prev_embed = tracker.node_prev_embeddings[0]
        
        # 计算相似性
        current_norm = current_embed / (torch.norm(current_embed, p=2) + 1e-8)
        prev_norm = prev_embed / (torch.norm(prev_embed, p=2) + 1e-8)
        similarity = torch.dot(current_norm, prev_norm).item()
        
        print(f"节点0的相似性: {similarity:.4f} (阈值: {tracker.similarity_threshold})")
        print(f"应该收敛: {similarity >= tracker.similarity_threshold}")
    
    tracker.update_node_info(node_ids, losses3, gradients3, embeddings3, 2)
    
    print(f"更新后:")
    print(f"- update_count: {tracker.node_update_count[node_ids].tolist()}")
    print(f"- eligible_nodes数量: {len(tracker.eligible_nodes)}")
    print(f"- converged_nodes数量: {len(tracker.converged_nodes)}")
    
    # 检查具体的相似性值
    if len(tracker.eligible_nodes) > 0:
        print("\n=== 检查eligible节点的相似性 ===")
        eligible_list = list(tracker.eligible_nodes)[:5]  # 取前5个
        for node_id in eligible_list:
            current_embed = tracker.node_last_embeddings[node_id]
            prev_embed = tracker.node_prev_embeddings[node_id]
            
            if current_embed.numel() > 0 and prev_embed.numel() > 0:
                current_norm = current_embed / (torch.norm(current_embed, p=2) + 1e-8)
                prev_norm = prev_embed / (torch.norm(prev_embed, p=2) + 1e-8)
                similarity = torch.dot(current_norm, prev_norm).item()
                
                print(f"节点{node_id}: 相似性={similarity:.4f}, 收敛={similarity >= tracker.similarity_threshold}")
    
    return tracker

def test_manual_convergence():
    """手动测试收敛逻辑"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n=== 手动测试收敛逻辑 ===")
    
    # 创建两个相似的向量
    embed_dim = 16
    embed1 = torch.randn(embed_dim, device=device)
    embed2 = embed1 + torch.randn(embed_dim, device=device) * 0.01  # 很小的噪音
    
    # 计算相似性
    embed1_norm = embed1 / (torch.norm(embed1, p=2) + 1e-8)
    embed2_norm = embed2 / (torch.norm(embed2, p=2) + 1e-8)
    similarity = torch.dot(embed1_norm, embed2_norm).item()
    
    print(f"手动计算的相似性: {similarity:.6f}")
    
    # 使用批量函数计算
    from info_feedback_system import compute_cosine_similarity_batch
    batch_sim = compute_cosine_similarity_batch(
        embed1.unsqueeze(0), embed2.unsqueeze(0)
    )[0].item()
    
    print(f"批量函数计算的相似性: {batch_sim:.6f}")
    print(f"两者是否一致: {abs(similarity - batch_sim) < 1e-5}")
    
    # 测试不同阈值
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    for threshold in thresholds:
        converged = similarity >= threshold
        print(f"阈值{threshold}: {'收敛' if converged else '未收敛'}")

if __name__ == "__main__":
    # 调试跟踪器
    tracker = debug_convergence_tracker()
    
    # 手动测试收敛逻辑
    test_manual_convergence()