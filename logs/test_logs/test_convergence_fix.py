#!/usr/bin/env python3
"""
测试修复后的收敛检测功能
"""
import torch
import time
from info_feedback_system import InfoFeedbackSystem

def test_convergence_detection():
    """测试收敛检测是否正常工作"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建系统 - 使用更宽松的参数
    num_nodes = 1000
    system = InfoFeedbackSystem(
        num_nodes=num_nodes,
        device=device,
        similarity_threshold=0.80,  # 降低到80%
        patience=2,  # 减少patience
        min_epochs=1,  # 最小epoch降为1
        use_simple_convergence=True
    )
    
    print(f"测试配置:")
    print(f"- 节点数: {num_nodes}")
    print(f"- 相似性阈值: 0.80")
    print(f"- 耐心值: 2")
    print(f"- 最小epochs: 1")
    print()
    
    # 模拟训练数据
    batch_size = 100
    embed_dim = 64
    
    # 第一轮：随机嵌入
    print("=== 第一轮：初始化随机嵌入 ===")
    node_ids = torch.arange(batch_size, device=device)
    losses = torch.randn(batch_size, device=device)
    gradients = torch.randn(batch_size, device=device) 
    embeddings = torch.randn(batch_size, embed_dim, device=device)
    
    feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, 0)
    print(f"Epoch 0: {feedback_info['converged_nodes_count']} nodes converged")
    
    # 第二轮：稍微变化的嵌入
    print("=== 第二轮：小幅变化嵌入 ===")
    embeddings = embeddings + torch.randn_like(embeddings) * 0.1  # 小幅变化
    feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, 1)
    print(f"Epoch 1: {feedback_info['converged_nodes_count']} nodes converged")
    
    # 第三轮：相似嵌入（模拟收敛）
    print("=== 第三轮：模拟收敛（相似嵌入）===")
    embeddings = embeddings + torch.randn_like(embeddings) * 0.05  # 更小变化
    feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, 2)
    print(f"Epoch 2: {feedback_info['converged_nodes_count']} nodes converged")
    
    # 第四轮：几乎不变的嵌入（应该触发收敛）
    print("=== 第四轮：几乎不变嵌入（应该收敛）===")
    embeddings = embeddings + torch.randn_like(embeddings) * 0.01  # 很小变化
    feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, 3)
    print(f"Epoch 3: {feedback_info['converged_nodes_count']} nodes converged")
    
    # 第五轮：完全相同的嵌入
    print("=== 第五轮：完全相同嵌入 ===")
    feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, 4)
    print(f"Epoch 4: {feedback_info['converged_nodes_count']} nodes converged")
    
    # 测试自适应采样
    print("\n=== 测试自适应采样 ===")
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[:num_nodes//2] = True
    
    for epoch in range(5, 10):
        batch_nodes = system.get_adaptive_batch(50, epoch, train_mask)
        print(f"Epoch {epoch}: 采样了 {len(batch_nodes)} 个节点")
    
    # 最终状态
    print(f"\n=== 最终系统状态 ===")
    final_status = system.get_status()
    print(f"最终状态: {final_status}")
    
    # 判断测试是否成功
    if final_status['converged_nodes_count'] > 0:
        print("✅ 测试成功：系统能够检测到节点收敛！")
        return True
    else:
        print("❌ 测试失败：系统未能检测到节点收敛")
        return False

def test_similarity_thresholds():
    """测试不同相似性阈值的效果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 测试不同相似性阈值 ===")
    
    thresholds = [0.70, 0.80, 0.90, 0.95]
    results = {}
    
    for threshold in thresholds:
        print(f"\n测试阈值: {threshold}")
        
        system = InfoFeedbackSystem(
            num_nodes=100,
            device=device,
            similarity_threshold=threshold,
            patience=2,
            min_epochs=1,
            use_simple_convergence=True
        )
        
        # 生成逐渐收敛的嵌入序列
        base_embedding = torch.randn(50, 32, device=device)
        node_ids = torch.arange(50, device=device)
        losses = torch.randn(50, device=device)
        gradients = torch.randn(50, device=device)
        
        converged_count = 0
        for epoch in range(5):
            # 逐渐减少变化，模拟收敛
            noise_scale = 0.5 * (0.5 ** epoch)  # 指数衰减
            embeddings = base_embedding + torch.randn_like(base_embedding) * noise_scale
            
            feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, epoch)
            converged_count = feedback_info['converged_nodes_count']
            if converged_count > 0:
                break
        
        results[threshold] = converged_count
        print(f"阈值 {threshold}: {converged_count} 个节点收敛")
    
    print(f"\n结果summary: {results}")
    return results

if __name__ == "__main__":
    print("=== 收敛检测修复测试 ===")
    print(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()
    
    # 测试基本收敛检测
    success = test_convergence_detection()
    print()
    
    # 测试不同阈值
    threshold_results = test_similarity_thresholds()
    print()
    
    if success:
        print("🎉 收敛检测功能修复成功！")
        print("建议的配置:")
        print("- similarity_threshold: 0.80-0.85")
        print("- patience: 2-3")  
        print("- min_epochs: 1")
    else:
        print("⚠️  收敛检测可能需要进一步调整")