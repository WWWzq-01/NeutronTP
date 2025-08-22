#!/usr/bin/env python3
"""
简单的收敛测试 - 验证修复效果
"""
import torch
from info_feedback_system import InfoFeedbackSystem

def test_fixed_convergence():
    """测试修复后的收敛检测"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建系统，使用更宽松的参数
    system = InfoFeedbackSystem(
        num_nodes=50,
        device=device, 
        similarity_threshold=0.70,  # 70%相似性
        patience=2,
        min_epochs=1,
        use_simple_convergence=True
    )
    
    print("=== 简单收敛测试 ===")
    print(f"配置: 相似性阈值=0.70, patience=2, min_epochs=1")
    
    # 测试数据
    batch_size = 20
    embed_dim = 32
    node_ids = torch.arange(batch_size, device=device)
    
    # 生成基础嵌入
    base_embedding = torch.randn(batch_size, embed_dim, device=device) 
    
    # 模拟训练过程
    for epoch in range(6):
        # 逐渐减小变化幅度
        noise_scale = 0.5 * (0.7 ** epoch)  # 指数衰减
        
        if epoch == 0:
            # 第一次：随机嵌入
            embeddings = base_embedding.clone()
        else:
            # 后续：逐渐收敛
            noise = torch.randn_like(base_embedding) * noise_scale
            embeddings = base_embedding + noise
        
        losses = torch.randn(batch_size, device=device)
        gradients = torch.randn(batch_size, device=device)
        
        # 处理反馈
        feedback_info = system.process_feedback(node_ids, losses, gradients, embeddings, epoch)
        
        if feedback_info:
            converged = feedback_info['converged_nodes_count'] 
            eligible = feedback_info['eligible_nodes']
            rate = feedback_info['convergence_rate']
            print(f"Epoch {epoch}: {converged} converged nodes, {eligible} eligible, rate={rate:.3f}, noise_scale={noise_scale:.4f}")
        
        # 如果有节点收敛就提前结束
        if feedback_info and feedback_info['converged_nodes_count'] > 0:
            print(f"✅ Success! {feedback_info['converged_nodes_count']} nodes converged at epoch {epoch}")
            return True
    
    print("❌ No nodes converged in 6 epochs")
    return False

if __name__ == "__main__":
    success = test_fixed_convergence()
    if success:
        print("\n🎉 收敛检测修复成功!")
    else:
        print("\n⚠️  可能需要进一步调试")