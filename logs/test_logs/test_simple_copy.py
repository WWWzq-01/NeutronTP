#!/usr/bin/env python3
"""
测试最简单的copy_操作
"""
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 最简单的测试
print("=== 最简单的copy_测试 ===")
target = torch.zeros(3, 4, device=device)
source = torch.randn(3, 4, device=device)

print(f"source: {source}")
print(f"target before copy: {target}")

target.copy_(source)

print(f"target after copy: {target}")

# 测试索引copy_
print("\n=== 索引copy_测试 ===")
target2 = torch.zeros(5, 4, device=device)
source2 = torch.randn(2, 4, device=device)
indices = torch.tensor([0, 2], device=device)

print(f"source2: {source2}")
print(f"target2 before: {target2}")
print(f"indices: {indices}")

target2[indices].copy_(source2)

print(f"target2 after: {target2}")

# 测试与NodeConvergenceTracker相同的操作
print("\n=== 模拟NodeConvergenceTracker操作 ===")
node_last_embeddings = torch.zeros(5, 4, device=device)
node_ids = torch.tensor([0, 1], device=device)
embeddings = torch.randn(2, 4, device=device)
node_update_count = torch.zeros(5, dtype=torch.int, device=device)

first_update_mask = node_update_count[node_ids] == 0
print(f"embeddings: {embeddings}")
print(f"first_update_mask: {first_update_mask}")

if first_update_mask.any():
    first_nodes = node_ids[first_update_mask]
    first_embeddings = embeddings[first_update_mask]
    
    print(f"first_nodes: {first_nodes}")
    print(f"first_embeddings: {first_embeddings}")
    print(f"node_last_embeddings[first_nodes] before: {node_last_embeddings[first_nodes]}")
    
    # 这是原来的操作
    node_last_embeddings[first_nodes].copy_(first_embeddings.detach())
    
    print(f"node_last_embeddings[first_nodes] after: {node_last_embeddings[first_nodes]}")
    print(f"node_last_embeddings: {node_last_embeddings}")

print("\n=== 检查内存共享问题 ===")
# 可能是内存共享问题？
print(f"embeddings.is_contiguous(): {embeddings.is_contiguous()}")
print(f"first_embeddings.is_contiguous(): {first_embeddings.is_contiguous()}")
print(f"node_last_embeddings[first_nodes].is_contiguous(): {node_last_embeddings[first_nodes].is_contiguous()}")

# 试试直接赋值
print(f"\n=== 直接赋值测试 ===")
node_last_embeddings2 = torch.zeros(5, 4, device=device)
node_last_embeddings2[first_nodes] = first_embeddings.detach()
print(f"直接赋值后: {node_last_embeddings2[first_nodes]}")