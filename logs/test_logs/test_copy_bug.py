#!/usr/bin/env python3
"""
测试copy_的bug
"""
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建存储矩阵
num_nodes = 10
embed_dim = 16
node_last_embeddings = torch.zeros((num_nodes, embed_dim), device=device)
node_update_count = torch.zeros(num_nodes, dtype=torch.int, device=device)

print("=== 测试copy_操作 ===")
print(f"初始 node_last_embeddings[0]: {node_last_embeddings[0]}")

# 模拟第一次更新
node_ids = torch.tensor([0, 1, 2], device=device)
embeddings = torch.randn(3, embed_dim, device=device)
first_update_mask = node_update_count[node_ids] == 0

print(f"输入 embeddings[0]: {embeddings[0]}")
print(f"first_update_mask: {first_update_mask}")

if first_update_mask.any():
    first_nodes = node_ids[first_update_mask]
    print(f"first_nodes: {first_nodes}")
    print(f"embeddings[first_update_mask]: {embeddings[first_update_mask]}")
    print(f"embeddings[first_update_mask].detach(): {embeddings[first_update_mask].detach()}")
    
    # 测试copy_操作
    print("执行copy_操作...")
    node_last_embeddings[first_nodes].copy_(embeddings[first_update_mask].detach())
    
    print(f"copy_后 node_last_embeddings[0]: {node_last_embeddings[0]}")

# 更新计数
node_update_count[node_ids] = node_update_count[node_ids] + 1
print(f"更新后 node_update_count[0]: {node_update_count[0]}")

print("\n=== 测试第二次更新 ===")
embeddings2 = torch.randn(3, embed_dim, device=device)
first_update_mask2 = node_update_count[node_ids] == 0

print(f"第二次输入 embeddings2[0]: {embeddings2[0]}")
print(f"第二次 first_update_mask: {first_update_mask2}")

if (~first_update_mask2).any():
    update_nodes = node_ids[~first_update_mask2]
    print(f"update_nodes: {update_nodes}")
    print(f"执行第二次更新前 node_last_embeddings[0]: {node_last_embeddings[0]}")
    
    # 这应该把当前的last保存到prev，然后更新last
    node_prev_embeddings = torch.zeros((num_nodes, embed_dim), device=device)
    node_prev_embeddings[update_nodes].copy_(node_last_embeddings[update_nodes])
    node_last_embeddings[update_nodes].copy_(embeddings2[~first_update_mask2].detach())
    
    print(f"第二次更新后 node_prev_embeddings[0]: {node_prev_embeddings[0]}")
    print(f"第二次更新后 node_last_embeddings[0]: {node_last_embeddings[0]}")