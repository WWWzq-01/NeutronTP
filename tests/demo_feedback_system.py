#!/usr/bin/env python3
"""
信息互馈系统演示脚本
展示如何使用信息互馈系统进行GNN训练
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from info_feedback_system import InfoFeedbackSystem, NodeConvergenceTracker, AdaptiveSampler
from feedback_config import ConfigManager, FeedbackConfigTemplates
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_synthetic_data(num_nodes=1000, num_features=64, num_classes=7):
    """创建合成数据用于演示"""
    print(f"创建合成数据: {num_nodes} 节点, {num_features} 特征, {num_classes} 类别")
    
    # 创建随机特征
    features = torch.randn(num_nodes, num_features)
    
    # 创建随机标签
    labels = torch.randint(0, num_classes, (num_nodes,))
    
    # 创建训练掩码
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_indices = torch.randperm(num_nodes)[:int(0.8 * num_nodes)]
    train_mask[train_indices] = True
    
    # 创建验证掩码
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_indices = torch.randperm(num_nodes)[int(0.8 * num_nodes):int(0.9 * num_nodes)]
    val_mask[val_indices] = True
    
    # 创建测试掩码
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_indices = torch.randperm(num_nodes)[int(0.9 * num_nodes):]
    test_mask[test_indices] = True
    
    return features, labels, train_mask, val_mask, test_mask

def simulate_training_epoch(features, labels, train_mask, model, optimizer, loss_func):
    """模拟一个训练epoch"""
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(features)
    
    # 计算损失
    loss = loss_func(outputs[train_mask], labels[train_mask])
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    return outputs, loss

def demo_basic_feedback_system():
    """演示基本的信息互馈系统"""
    print("\n" + "="*60)
    print("演示1: 基本信息互馈系统")
    print("="*60)
    
    # 创建合成数据
    num_nodes = 1000
    features, labels, train_mask, val_mask, test_mask = create_synthetic_data(num_nodes)
    
    # 创建信息互馈系统
    device = torch.device('cpu')
    feedback_system = InfoFeedbackSystem(num_nodes, device, enable_feedback=True)
    
    print(f"信息互馈系统已创建，节点数: {num_nodes}")
    print(f"训练节点数: {train_mask.sum().item()}")
    print(f"验证节点数: {val_mask.sum().item()}")
    print(f"测试节点数: {test_mask.sum().item()}")
    
    # 模拟训练过程
    num_epochs = 20
    batch_size = 200
    
    print(f"\n开始模拟训练，共 {num_epochs} 个epoch，batch大小: {batch_size}")
    
    for epoch in range(num_epochs):
        # 获取自适应batch
        batch_nodes = feedback_system.get_adaptive_batch(batch_size, epoch, train_mask)
        
        # 模拟训练反馈
        batch_features = features[batch_nodes]
        batch_labels = labels[batch_nodes]
        
        # 模拟损失和梯度
        batch_losses = torch.rand(len(batch_nodes)) * (1.0 - epoch * 0.05)  # 损失逐渐减小
        batch_gradients = torch.randn(len(batch_nodes), features.shape[1]) * (1.0 - epoch * 0.05)
        batch_embeddings = torch.randn(len(batch_nodes), features.shape[1])
        
        # 处理反馈信息
        feedback_info = feedback_system.process_feedback(
            batch_nodes, batch_losses, batch_gradients, batch_embeddings, epoch
        )
        
        # 调整策略
        if feedback_info:
            convergence_rate = feedback_info['convergence_rate']
            feedback_system.adjust_strategy(epoch, convergence_rate)
            
            print(f"Epoch {epoch:2d}: 收敛率 {convergence_rate:.3f}, "
                  f"收敛节点数 {feedback_info['converged_nodes_count']}")
        
        # 每5个epoch打印系统状态
        if epoch % 5 == 0:
            status = feedback_system.get_status()
            print(f"  Epoch {epoch} 系统状态: {status}")
    
    # 打印最终状态
    final_status = feedback_system.get_status()
    print(f"\n训练完成，最终状态: {final_status}")

def demo_sampling_strategies():
    """演示不同的采样策略"""
    print("\n" + "="*60)
    print("演示2: 不同采样策略对比")
    print("="*60)
    
    num_nodes = 500
    features, labels, train_mask, val_mask, test_mask = create_synthetic_data(num_nodes)
    
    device = torch.device('cpu')
    
    # 测试不同的采样策略
    strategies = ['adaptive_importance', 'convergence_aware', 'random']
    
    for strategy in strategies:
        print(f"\n测试采样策略: {strategy}")
        
        # 创建收敛跟踪器
        convergence_tracker = NodeConvergenceTracker(num_nodes)
        
        # 创建采样器
        sampler = AdaptiveSampler(num_nodes, convergence_tracker, strategy)
        
        # 模拟一些节点收敛
        for i in range(100):
            convergence_tracker.converged_nodes.add(i)
        
        print(f"  已收敛节点数: {len(convergence_tracker.converged_nodes)}")
        
        # 测试采样
        batch_size = 100
        epoch = 10
        
        for test_run in range(3):
            sampled_nodes = sampler.sample_nodes(batch_size, epoch, train_mask)
            converged_in_batch = sum(1 for n in sampled_nodes if n.item() in convergence_tracker.converged_nodes)
            
            print(f"  测试 {test_run + 1}: 采样 {len(sampled_nodes)} 节点, "
                  f"其中收敛节点 {converged_in_batch} 个")

def demo_config_management():
    """演示配置管理"""
    print("\n" + "="*60)
    print("演示3: 配置管理")
    print("="*60)
    
    # 测试不同的配置模板
    config_names = ['balanced', 'conservative', 'aggressive', 'memory_optimized']
    
    for config_name in config_names:
        print(f"\n配置模板: {config_name}")
        try:
            config_manager = ConfigManager(config_name)
            config = config_manager.get_config()
            
            print(f"  收敛阈值: {config['convergence']['threshold']}")
            print(f"  探索率: {config['sampling']['exploration_rate']}")
            print(f"  收敛惩罚: {config['sampling']['convergence_penalty']}")
            
        except Exception as e:
            print(f"  错误: {e}")
    
    # 测试配置验证
    print(f"\n配置验证测试:")
    try:
        # 创建一个无效配置
        invalid_config = FeedbackConfigTemplates.balanced()
        invalid_config['convergence']['threshold'] = -0.1  # 无效值
        
        errors = ConfigValidator.validate_all_config(invalid_config)
        if errors:
            print(f"  配置验证失败: {errors}")
        else:
            print("  配置验证通过")
            
    except Exception as e:
        print(f"  配置验证异常: {e}")

def demo_performance_comparison():
    """演示性能对比"""
    print("\n" + "="*60)
    print("演示4: 性能对比分析")
    print("="*60)
    
    num_nodes = 2000
    features, labels, train_mask, val_mask, test_mask = create_synthetic_data(num_nodes)
    
    device = torch.device('cpu')
    
    # 对比启用和禁用反馈系统的性能
    results = {}
    
    for enable_feedback in [False, True]:
        print(f"\n测试 {'启用' if enable_feedback else '禁用'} 反馈系统")
        
        feedback_system = InfoFeedbackSystem(num_nodes, device, enable_feedback)
        
        # 模拟训练过程
        num_epochs = 15
        batch_size = 300
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        convergence_rates = []
        
        for epoch in range(num_epochs):
            # 获取batch
            batch_nodes = feedback_system.get_adaptive_batch(batch_size, epoch, train_mask)
            
            # 模拟训练反馈
            batch_losses = torch.rand(len(batch_nodes)) * (1.0 - epoch * 0.06)
            batch_gradients = torch.randn(len(batch_nodes), features.shape[1]) * (1.0 - epoch * 0.06)
            batch_embeddings = torch.randn(len(batch_nodes), features.shape[1])
            
            if enable_feedback:
                feedback_info = feedback_system.process_feedback(
                    batch_nodes, batch_losses, batch_gradients, batch_embeddings, epoch
                )
                if feedback_info:
                    convergence_rates.append(feedback_info['convergence_rate'])
                    feedback_system.adjust_strategy(epoch, feedback_info['convergence_rate'])
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
        else:
            elapsed_time = 0
        
        results[f'feedback_{enable_feedback}'] = {
            'elapsed_time': elapsed_time,
            'final_convergence_rate': convergence_rates[-1] if convergence_rates else 0,
            'avg_convergence_rate': np.mean(convergence_rates) if convergence_rates else 0
        }
        
        print(f"  训练时间: {elapsed_time:.2f} ms")
        print(f"  最终收敛率: {results[f'feedback_{enable_feedback}']['final_convergence_rate']:.3f}")
        print(f"  平均收敛率: {results[f'feedback_{enable_feedback}']['avg_convergence_rate']:.3f}")
    
    # 性能分析
    print(f"\n性能对比分析:")
    if results['feedback_True']['elapsed_time'] > 0 and results['feedback_False']['elapsed_time'] > 0:
        time_ratio = results['feedback_True']['elapsed_time'] / results['feedback_False']['elapsed_time']
        print(f"  时间开销比例: {time_ratio:.2f}x")
    
    convergence_improvement = (results['feedback_True']['final_convergence_rate'] - 
                              results['feedback_False']['final_convergence_rate'])
    print(f"  收敛率提升: {convergence_improvement:.3f}")

def plot_convergence_comparison():
    """绘制收敛对比图"""
    print("\n" + "="*60)
    print("演示5: 收敛曲线可视化")
    print("="*60)
    
    # 模拟训练过程数据
    epochs = list(range(1, 21))
    
    # 标准训练（无反馈）
    standard_convergence = [0.1 + 0.04 * epoch for epoch in epochs]
    
    # 信息互馈训练
    feedback_convergence = [0.1 + 0.06 * epoch + 0.002 * epoch**2 for epoch in epochs]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 收敛率对比
    plt.subplot(2, 2, 1)
    plt.plot(epochs, standard_convergence, 'b-', label='标准训练', linewidth=2)
    plt.plot(epochs, feedback_convergence, 'r-', label='信息互馈训练', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('收敛率')
    plt.title('收敛率对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 收敛率提升
    plt.subplot(2, 2, 2)
    improvement = [f - s for f, s in zip(feedback_convergence, standard_convergence)]
    plt.plot(epochs, improvement, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('收敛率提升')
    plt.title('信息互馈带来的收敛率提升')
    plt.grid(True, alpha=0.3)
    
    # 节点重要性分布
    plt.subplot(2, 2, 3)
    node_importance = np.random.exponential(0.5, 100)
    plt.hist(node_importance, bins=20, alpha=0.7, color='orange')
    plt.xlabel('节点重要性分数')
    plt.ylabel('节点数量')
    plt.title('节点重要性分布')
    plt.grid(True, alpha=0.3)
    
    # 采样策略效果
    plt.subplot(2, 2, 4)
    strategies = ['随机采样', '重要性采样', '收敛感知采样']
    performance = [0.65, 0.78, 0.82]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = plt.bar(strategies, performance, color=colors, alpha=0.8)
    plt.ylabel('性能分数')
    plt.title('不同采样策略性能对比')
    plt.ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for bar, perf in zip(bars, performance):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{perf:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feedback_system_analysis.png', dpi=300, bbox_inches='tight')
    print("  图表已保存为 'feedback_system_analysis.png'")
    plt.show()

def main():
    """主函数"""
    print("信息互馈系统演示")
    print("="*60)
    
    try:
        # 运行所有演示
        demo_basic_feedback_system()
        demo_sampling_strategies()
        demo_config_management()
        demo_performance_comparison()
        plot_convergence_comparison()
        
        print("\n" + "="*60)
        print("所有演示完成！")
        print("="*60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
