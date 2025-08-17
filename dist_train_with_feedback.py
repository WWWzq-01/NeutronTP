from coo_graph import Parted_COO_Graph
from coo_graph import Full_COO_Graph
from coo_graph import Full_COO_Graph_Large
from coo_graph import Full_COO_Graph_CPU
from models import GCN, GAT, CachedGCN, DecoupleGCN, TensplitGCN, TensplitGCNLARGE, TensplitGCNCPU, TensplitGAT
from info_feedback_system import InfoFeedbackSystem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from dist_utils import DistEnv
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def f1(y_true, y_pred, multilabel=True):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if multilabel:
        # 对预测值进行二值化
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0
        # 在一些节点（10、100、1000）上打印预测值和真实值
        for node in [10,100,1000]:
            DistEnv.env.logger.log('pred', y_pred[node] , rank=0)
            DistEnv.env.logger.log('true', y_true[node] , rank=0)
    else:
        # 如果是单标签分类，将预测值转为类别索引
        y_pred = np.argmax(y_pred, axis=1)
    # 返回 micro 和 macro 两种平均方式的 F1 分数
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")

def train_with_feedback(g, env, args, feedback_system: InfoFeedbackSystem):
    """集成信息互馈的训练函数"""
    if args.model == 'GCN':
        model = GCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'CachedGCN':
        model = CachedGCN(g, env, hidden_dim=args.hidden)
    elif args.model == 'GAT':
        model = GAT(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'DecoupleGCN':
        model = DecoupleGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCN':
        model = TensplitGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCNLARGE':
        model = TensplitGCNLARGE(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCNCPU':
        model = TensplitGCNCPU(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGAT':
        model = TensplitGAT(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)

    # 创建优化器（Adam）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        # 对于单标签分类，使用交叉熵损失函数
        loss_func = nn.CrossEntropyLoss(reduction='none')  # 改为none以获取每个节点的损失
    elif g.labels.dim()==2:
        # 对于多标签分类，使用 BCEWithLogitsLoss 损失函数
        loss_func = nn.BCEWithLogitsLoss(reduction='none')  # 改为none以获取每个节点的损失
    
    # 训练循环
    for epoch in range(args.epoch):
        with env.timer.timing('epoch'):
            with autocast(env.half_enabled):
                # 前向传播，计算输出
                outputs = model(g.features)
                
                # 梯度清零
                optimizer.zero_grad()
                
                if g.local_labels[g.local_train_mask].size(0) > 0:
                    # 获取训练节点
                    train_nodes = torch.where(g.local_train_mask)[0]
                    
                    # 使用信息互馈系统获取自适应batch
                    if feedback_system.is_enabled():
                        batch_size = min(args.batch_size if hasattr(args, 'batch_size') else len(train_nodes), 
                                       len(train_nodes))
                        batch_nodes = feedback_system.get_adaptive_batch(batch_size, epoch, g.local_train_mask)
                        
                        # 边界情况处理：如果batch为空，跳过反馈处理
                        if len(batch_nodes) == 0:
                            env.logger.log(f"Warning: Empty batch at epoch {epoch}, skipping feedback processing", rank=0)
                            # 使用所有训练节点进行标准训练
                            batch_nodes = train_nodes
                            loss = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes]).mean()
                            loss.backward()
                        else:
                            # 计算batch上的损失
                            if g.labels.dim() == 1:
                                batch_losses = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes])
                            else:
                                batch_losses = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes]).mean(dim=1)
                            
                            # 计算总损失
                            loss = batch_losses.mean()
                            
                            # 处理信息互馈
                            # 获取节点梯度信息 - 使用简化的方法
                            batch_outputs = outputs[batch_nodes]
                            
                            # 计算梯度
                            loss.backward(retain_graph=True)
                            
                            # 使用损失值作为梯度的替代指标
                            # 这样可以避免复杂的梯度获取逻辑
                            node_gradients = batch_losses.detach().clone()
                            
                            # 处理反馈信息
                            feedback_info = feedback_system.process_feedback(
                                batch_nodes, batch_losses.detach(), 
                                node_gradients, batch_outputs.detach(), epoch)
                            
                            # 根据反馈调整策略
                            if feedback_info:
                                convergence_rate = feedback_info['convergence_rate']
                                feedback_system.adjust_strategy(epoch, convergence_rate)
                                
                                # 记录反馈信息
                                env.logger.log(f"Epoch {epoch} | Feedback: {convergence_rate:.3f} nodes converged", rank=0)
                            
                            # 重新计算损失（因为retain_graph=True会消耗梯度）
                            optimizer.zero_grad()
                            loss = batch_losses.mean()
                            loss.backward()
                    else:
                        # 如果没有启用反馈系统，使用所有训练节点
                        batch_nodes = train_nodes
                        loss = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes]).mean()
                        loss.backward()
                
                # 参数更新
                optimizer.step()
                
                # 输出当前的损失信息
                env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)

        # 每10个epoch或最后一个epoch进行评估
        if epoch%10==0 or epoch==args.epoch-1:
            # 收集所有节点的输出，并拼接在一起
            all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                # 如果是多标签分类，计算 F1 分数并打印
                mask = g.train_mask
                env.logger.log(f'Epoch: {epoch:03d}', f1(g.labels[mask], torch.sigmoid(all_outputs[mask])), rank=0)
            else:
                # 如果是单标签分类，计算并打印训练/验证/测试的准确率
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)
        
        # 打印信息互馈系统状态
        if feedback_system.is_enabled() and epoch % 5 == 0:
            status = feedback_system.get_status()
            env.logger.log(f"Epoch {epoch} | Feedback System Status: {status}", rank=0)


def train(g, env, args):
    """原始训练函数（保持兼容性）"""
    if args.model == 'GCN':
        model = GCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'CachedGCN':
        model = CachedGCN(g, env, hidden_dim=args.hidden)
    elif args.model == 'GAT':
        model = GAT(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'DecoupleGCN':
        model = DecoupleGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCN':
        model = TensplitGCN(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCNLARGE':
        model = TensplitGCNLARGE(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGCNCPU':
        model = TensplitGCNCPU(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)
    elif args.model == 'TensplitGAT':
        model = TensplitGAT(g, env, hidden_dim=args.hidden, nlayers=args.nlayers)

    # 创建优化器（Adam）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        # 对于单标签分类，使用交叉熵损失函数
        loss_func = nn.CrossEntropyLoss()
    elif g.labels.dim()==2:
        # 对于多标签分类，使用 BCEWithLogitsLoss 损失函数
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    
    for epoch in range(args.epoch):
        with env.timer.timing('epoch'):
            with autocast(env.half_enabled):
                # 前向传播，计算输出
                outputs = model(g.features)
                # 梯度清零
                optimizer.zero_grad()
                if g.local_labels[g.local_train_mask].size(0) > 0:
                    # 计算损失（仅在包含训练节点的分区上计算）
                    loss = loss_func(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
                else:
                    # 如果没有训练节点，输出警告并使用虚拟损失
                    env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
                    loss = (outputs * 0).sum()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
            # 输出当前的损失信息
            env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)

        if epoch%10==0 or epoch==args.epoch-1:
            # 收集所有节点的输出，并拼接在一起
            all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                # 如果是多标签分类，计算 F1 分数并打印
                mask = g.train_mask
                env.logger.log(f'Epoch: {epoch:03d}', f1(g.labels[mask], torch.sigmoid(all_outputs[mask])), rank=0)
            else:
                # 如果是单标签分类，计算并打印训练/验证/测试的准确率
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)


def main(env, args):
    env.csr_enabled = False
    env.csr_enabled = True

    env.half_enabled = True
    env.half_enabled = False
    
    # 打印进程开始信息
    env.logger.log('proc begin:', env)
    
    with env.timer.timing('total'):
        # 使用 Parted_COO_Graph 加载分布式环境下的图数据
        print("world size",env.world_size)
        if args.model == 'TensplitGCN':
            print(f"Rank: {env.rank}, world_size: {env.world_size}")
            g = Full_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        elif args.model == 'TensplitGAT':
            env.csr_enabled = False
            g = Full_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        elif args.model == 'TensplitGCNCPU':
            g = Full_COO_Graph_CPU(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        elif args.model == 'TensplitGCNLARGE':
            g = Full_COO_Graph_Large(args.dataset, env.rank, env.world_size, args.chunk, env.device, env.half_enabled, env.csr_enabled)
        elif args.model == 'GAT':
            g = Parted_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        else:
            g = Parted_COO_Graph(args.dataset, env.rank, env.world_size, env.device, env.half_enabled, env.csr_enabled)
        
        env.logger.log('graph loaded', g)
        env.logger.log('graph loaded\n', torch.cuda.memory_summary())
        
        # 初始化信息互馈系统
        enable_feedback = getattr(args, 'enable_feedback', True)
        feedback_system = InfoFeedbackSystem(g.num_nodes, env.device, enable_feedback)
        
        # 根据是否启用反馈系统选择训练函数
        if enable_feedback:
            env.logger.log('Starting training with Info-Feedback system', rank=0)
            train_with_feedback(g, env, args, feedback_system)
        else:
            env.logger.log('Starting standard training without feedback', rank=0)
            train(g, env, args)
    
    # 打印model信息
    if env.rank == 0:    
        print(f"Model: {args.model} layers: {args.nlayers} nprocs {args.nprocs}")
        if enable_feedback:
            print("Info-Feedback system was enabled")
        else:
            print("Info-Feedback system was disabled")
    
    # 打印计时器的总结信息
    env.logger.log(env.timer.summary_all(), rank=0)
    
    # 打印信息互馈系统最终状态
    if enable_feedback:
        final_status = feedback_system.get_status()
        env.logger.log(f"Final Feedback System Status: {final_status}", rank=0)
