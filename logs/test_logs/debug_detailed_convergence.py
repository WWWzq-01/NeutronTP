#!/usr/bin/env python3
"""
详细调试收敛检测问题 - 追踪嵌入向量的具体值
"""
import torch
from info_feedback_system import NodeConvergenceTracker

def debug_embedding_values():
    """追踪嵌入向量的具体数值"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建简单的跟踪器
    tracker = NodeConvergenceTracker(
        num_nodes=10,
        patience=2,
        min_epochs=1,
        similarity_threshold=0.80,
        use_simple_convergence=True,
        device=device
    )
    
    print("=== 详细调试嵌入向量值 ===")
    print(f"设备: {device}")
    
    # 测试数据
    batch_size = 3
    embed_dim = 4  # 小维度便于观察
    node_ids = torch.tensor([0, 1, 2], device=device)
    
    # 第一次更新
    print("\n=== 第一次更新 ===")
    embeddings1 = torch.randn(batch_size, embed_dim, device=device)
    losses1 = torch.randn(batch_size, device=device)
    gradients1 = torch.randn(batch_size, device=device)
    
    print(f"输入嵌入 embeddings1[0]: {embeddings1[0]}")
    
    tracker.update_node_info(node_ids, losses1, gradients1, embeddings1, 0)
    
    print(f"存储后 node_last_embeddings[0]: {tracker.node_last_embeddings[0]}")
    print(f"存储后 node_prev_embeddings[0]: {tracker.node_prev_embeddings[0]}")
    print(f"update_count[0]: {tracker.node_update_count[0]}")
    print()
    
    # 第二次更新
    print("=== 第二次更新 ===")
    embeddings2 = embeddings1 + torch.randn_like(embeddings1) * 0.1
    losses2 = torch.randn(batch_size, device=device)
    gradients2 = torch.randn(batch_size, device=device)
    
    print(f"输入嵌入 embeddings2[0]: {embeddings2[0]}")
    print(f"更新前 node_last_embeddings[0]: {tracker.node_last_embeddings[0]}")
    print(f"更新前 node_prev_embeddings[0]: {tracker.node_prev_embeddings[0]}")
    
    tracker.update_node_info(node_ids, losses2, gradients2, embeddings2, 1)
    
    print(f"更新后 node_last_embeddings[0]: {tracker.node_last_embeddings[0]}")
    print(f"更新后 node_prev_embeddings[0]: {tracker.node_prev_embeddings[0]}")
    print(f"update_count[0]: {tracker.node_update_count[0]}")
    print()
    
    # 第三次更新（使用相同嵌入）
    print("=== 第三次更新（相同嵌入）===")
    embeddings3 = embeddings2.clone()  # 完全相同的嵌入
    losses3 = torch.randn(batch_size, device=device)
    gradients3 = torch.randn(batch_size, device=device)
    
    print(f"输入嵌入 embeddings3[0]: {embeddings3[0]}")
    print(f"更新前 node_last_embeddings[0]: {tracker.node_last_embeddings[0]}")
    print(f"更新前 node_prev_embeddings[0]: {tracker.node_prev_embeddings[0]}")
    
    # 手动计算预期的相似性
    if tracker.node_update_count[0] >= tracker.patience:
        current_embed = tracker.node_last_embeddings[0]
        prev_embed = tracker.node_prev_embeddings[0]
        
        print(f"手动计算相似性:")
        print(f"  current: {current_embed}")
        print(f"  prev: {prev_embed}")
        
        # 手动计算余弦相似性
        current_norm = current_embed / (torch.norm(current_embed, p=2) + 1e-8)
        prev_norm = prev_embed / (torch.norm(prev_embed, p=2) + 1e-8)
        similarity = torch.dot(current_norm, prev_norm).item()
        
        print(f"  current_norm: {current_norm}")
        print(f"  prev_norm: {prev_norm}")
        print(f"  手动相似性: {similarity:.6f}")
        
        # 检查两个向量是否真的相同
        print(f"  current == prev? {torch.equal(current_embed, prev_embed)}")
        print(f"  差值范数: {torch.norm(current_embed - prev_embed).item():.6f}")
    
    tracker.update_node_info(node_ids, losses3, gradients3, embeddings3, 2)
    
    print(f"更新后 node_last_embeddings[0]: {tracker.node_last_embeddings[0]}")
    print(f"更新后 node_prev_embeddings[0]: {tracker.node_prev_embeddings[0]}")
    print(f"update_count[0]: {tracker.node_update_count[0]}")
    print(f"eligible_nodes数量: {len(tracker.eligible_nodes)}")
    print(f"converged_nodes数量: {len(tracker.converged_nodes)}")
    
    # 最后手动验证相似性计算
    if len(tracker.eligible_nodes) > 0:
        print("\n=== 最终相似性验证 ===")
        current_embed = tracker.node_last_embeddings[0]
        prev_embed = tracker.node_prev_embeddings[0]
        
        similarity_manual = torch.dot(
            current_embed / (torch.norm(current_embed, p=2) + 1e-8),
            prev_embed / (torch.norm(prev_embed, p=2) + 1e-8)
        ).item()
        
        # 使用批量函数计算
        similarity_batch = tracker._compute_embedding_similarity_batch(
            current_embed.unsqueeze(0), prev_embed.unsqueeze(0)
        )[0].item()
        
        print(f"手动计算相似性: {similarity_manual:.6f}")
        print(f"批量函数相似性: {similarity_batch:.6f}")
        print(f"是否应该收敛: {similarity_batch >= tracker.similarity_threshold}")
        print(f"阈值: {tracker.similarity_threshold}")

if __name__ == "__main__":
    debug_embedding_values()