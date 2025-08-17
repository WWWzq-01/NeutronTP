import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict, deque
import logging

class NodeConvergenceTracker:
    """节点收敛状态跟踪器"""
    
    def __init__(self, num_nodes: int, convergence_threshold: float = 0.01, 
                 patience: int = 5, min_epochs: int = 3, embedding_dim: int = 64):
        self.num_nodes = num_nodes
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.min_epochs = min_epochs
        self.embedding_dim = embedding_dim  # 添加嵌入维度参数
        
        # 每个节点的历史损失和梯度信息
        self.node_loss_history = defaultdict(lambda: deque(maxlen=patience))
        self.node_grad_history = defaultdict(lambda: deque(maxlen=patience))
        self.node_embedding_history = defaultdict(lambda: deque(maxlen=patience))
        
        # 节点收敛状态
        self.converged_nodes = set()
        self.convergence_epochs = defaultdict(int)
        
        # 节点重要性分数
        self.node_importance_scores = torch.ones(num_nodes, dtype=torch.float32)
    
    def update_embedding_dim(self, new_dim: int):
        """更新嵌入维度"""
        self.embedding_dim = new_dim
    
    def update_node_info(self, node_ids: torch.Tensor, losses: torch.Tensor, 
                        gradients: torch.Tensor, embeddings: torch.Tensor, epoch: int):
        """更新节点的训练信息"""
        # 动态更新嵌入维度 - 修复：先检查张量是否为空
        if (embeddings is not None and 
            embeddings.numel() > 0 and  # 检查张量是否为空
            embeddings.dim() > 1 and    # 检查是否有多个维度
            embeddings.size(1) != self.embedding_dim):
            self.update_embedding_dim(embeddings.size(1))
        
        for i, node_id in enumerate(node_ids):
            node_id = node_id.item()
            if node_id < self.num_nodes:
                self.node_loss_history[node_id].append(losses[i].item())
                
                # 简化：使用损失值作为梯度信息的替代
                # 这样可以避免复杂的梯度计算问题
                if gradients is not None and i < gradients.size(0):
                    # 如果梯度是标量，直接使用；如果是向量，计算范数
                    if gradients[i].dim() == 0:
                        grad_value = gradients[i].item()
                    else:
                        grad_value = gradients[i].norm().item()
                    self.node_grad_history[node_id].append(grad_value)
                else:
                    # 如果没有梯度信息，使用损失值作为替代
                    self.node_grad_history[node_id].append(losses[i].item())
                
                # 安全处理嵌入信息
                if (embeddings is not None and 
                    embeddings.numel() > 0 and  # 检查张量是否为空
                    i < embeddings.size(0)):
                    self.node_embedding_history[node_id].append(embeddings[i].detach().clone())
                else:
                    # 如果没有嵌入信息，创建一个零向量
                    dummy_embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)
                    self.node_embedding_history[node_id].append(dummy_embedding)
                
                # 检查是否收敛
                if self._check_convergence(node_id, epoch):
                    self.converged_nodes.add(node_id)
                    self.convergence_epochs[node_id] = epoch
    
    def _check_convergence(self, node_id: int, epoch: int) -> bool:
        """检查节点是否收敛"""
        if epoch < self.min_epochs:
            return False
            
        if node_id in self.converged_nodes:
            return False
            
        loss_history = self.node_loss_history[node_id]
        grad_history = self.node_grad_history[node_id]
        embed_history = self.node_embedding_history[node_id]
        
        if len(loss_history) < self.patience:
            return False
            
        # 检查损失变化
        loss_variance = np.var(list(loss_history))
        if loss_variance > self.convergence_threshold:
            return False
            
        # 检查梯度变化
        grad_variance = np.var(list(grad_history))
        if grad_variance > self.convergence_threshold:
            return False
            
        # 检查嵌入变化
        if len(embed_history) >= 2:
            embed_diff = torch.norm(embed_history[-1] - embed_history[-2])
            if embed_diff > self.convergence_threshold:
                return False
                
        return True
    
    def get_converged_nodes(self) -> set:
        """获取已收敛的节点"""
        return self.converged_nodes.copy()
    
    def get_node_importance(self, node_ids: torch.Tensor) -> torch.Tensor:
        """获取节点的重要性分数"""
        return self.node_importance_scores[node_ids]
    
    def update_importance_scores(self, node_ids: torch.Tensor, 
                               new_scores: torch.Tensor):
        """更新节点重要性分数"""
        self.node_importance_scores[node_ids] = new_scores


class AdaptiveSampler:
    """自适应采样器，根据训练反馈调整采样策略"""
    
    def __init__(self, num_nodes: int, convergence_tracker: NodeConvergenceTracker,
                 sampling_strategy: str = 'adaptive_importance'):
        self.num_nodes = num_nodes
        self.convergence_tracker = convergence_tracker
        self.sampling_strategy = sampling_strategy
        
        # 采样统计
        self.sampling_counts = torch.zeros(num_nodes, dtype=torch.long)
        self.last_sampling_epoch = torch.full((num_nodes,), -1, dtype=torch.long)
        
        # 采样策略参数
        self.exploration_rate = 0.1  # 探索率
        self.convergence_penalty = 0.5  # 收敛节点惩罚因子
        
    def sample_nodes(self, batch_size: int, epoch: int, 
                    train_mask: torch.Tensor) -> torch.Tensor:
        """根据当前状态自适应采样节点"""
        if self.sampling_strategy == 'adaptive_importance':
            return self._adaptive_importance_sampling(batch_size, epoch, train_mask)
        elif self.sampling_strategy == 'convergence_aware':
            return self._convergence_aware_sampling(batch_size, epoch, train_mask)
        else:
            return self._random_sampling(batch_size, train_mask)
    
    def _adaptive_importance_sampling(self, batch_size: int, epoch: int, 
                                    train_mask: torch.Tensor) -> torch.Tensor:
        """基于重要性的自适应采样"""
        # 获取训练节点
        train_nodes = torch.where(train_mask)[0]
        
        # 边界情况处理：如果没有训练节点，返回空张量
        if len(train_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=train_mask.device)
        
        # 如果训练节点数小于等于batch_size，直接返回所有节点
        if len(train_nodes) <= batch_size:
            return train_nodes
        
        # 计算采样概率
        importance_scores = self.convergence_tracker.get_node_importance(train_nodes)
        
        # 对收敛节点应用惩罚
        converged_nodes = self.convergence_tracker.get_converged_nodes()
        for i, node_id in enumerate(train_nodes):
            if node_id.item() in converged_nodes:
                importance_scores[i] *= self.convergence_penalty
        
        # 添加探索性采样
        exploration_mask = torch.rand(len(train_nodes)) < self.exploration_rate
        importance_scores[exploration_mask] = 1.0
        
        # 归一化概率
        probs = importance_scores / importance_scores.sum()
        
        # 采样
        sampled_indices = torch.multinomial(probs, batch_size, replacement=False)
        sampled_nodes = train_nodes[sampled_indices]
        
        # 更新统计信息
        self.sampling_counts[sampled_nodes] += 1
        self.last_sampling_epoch[sampled_nodes] = epoch
        
        return sampled_nodes
    
    def _convergence_aware_sampling(self, batch_size: int, epoch: int,
                                   train_mask: torch.Tensor) -> torch.Tensor:
        """基于收敛状态的采样"""
        train_nodes = torch.where(train_mask)[0]
        
        # 边界情况处理：如果没有训练节点，返回空张量
        if len(train_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=train_mask.device)
        
        converged_nodes = self.convergence_tracker.get_converged_nodes()
        
        # 分离收敛和未收敛节点
        unconverged_nodes = [n for n in train_nodes if n.item() not in converged_nodes]
        converged_train_nodes = [n for n in train_nodes if n.item() in converged_nodes]
        
        # 优先采样未收敛节点
        if len(unconverged_nodes) >= batch_size:
            sampled_nodes = torch.tensor(unconverged_nodes[:batch_size])
        else:
            # 如果未收敛节点不够，补充一些收敛节点
            remaining_size = batch_size - len(unconverged_nodes)
            sampled_nodes = torch.cat([
                torch.tensor(unconverged_nodes),
                torch.tensor(converged_train_nodes[:remaining_size])
            ])
        
        # 更新统计信息
        self.sampling_counts[sampled_nodes] += 1
        self.last_sampling_epoch[sampled_nodes] = epoch
        
        return sampled_nodes
    
    def _random_sampling(self, batch_size: int, train_mask: torch.Tensor) -> torch.Tensor:
        """随机采样（基准方法）"""
        train_nodes = torch.where(train_mask)[0]
        
        # 边界情况处理：如果没有训练节点，返回空张量
        if len(train_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=train_mask.device)
        
        if len(train_nodes) <= batch_size:
            return train_nodes
        else:
            indices = torch.randperm(len(train_nodes))[:batch_size]
            return train_nodes[indices]


class FeedbackController:
    """反馈控制器，协调整个信息互馈系统"""
    
    def __init__(self, num_nodes: int, device: torch.device):
        self.device = device
        self.num_nodes = num_nodes
        
        # 初始化组件
        self.convergence_tracker = NodeConvergenceTracker(num_nodes)
        self.adaptive_sampler = AdaptiveSampler(num_nodes, self.convergence_tracker)
        
        # 反馈统计
        self.feedback_stats = {
            'total_feedback_cycles': 0,
            'nodes_converged': 0,
            'sampling_adaptations': 0
        }
        
        # 日志记录
        self.logger = logging.getLogger(__name__)
        
    def process_training_feedback(self, node_ids: torch.Tensor, losses: torch.Tensor,
                                gradients: torch.Tensor, embeddings: torch.Tensor, 
                                epoch: int) -> Dict:
        """处理训练反馈信息"""
        # 边界情况处理：如果输入为空，返回默认反馈信息
        if node_ids.numel() == 0:
            feedback_info = {
                'epoch': epoch,
                'converged_nodes_count': 0,
                'convergence_rate': 0.0,
                'total_nodes': self.num_nodes
            }
            self.logger.info(f"Epoch {epoch}: Empty batch, no feedback to process")
            return feedback_info
        
        # 更新节点收敛状态
        self.convergence_tracker.update_node_info(node_ids, losses, gradients, 
                                                embeddings, epoch)
        
        # 计算反馈统计
        converged_count = len(self.convergence_tracker.get_converged_nodes())
        feedback_info = {
            'epoch': epoch,
            'converged_nodes_count': converged_count,
            'convergence_rate': converged_count / self.num_nodes,
            'total_nodes': self.num_nodes
        }
        
        # 记录反馈统计
        self.feedback_stats['total_feedback_cycles'] += 1
        self.feedback_stats['nodes_converged'] = converged_count
        
        self.logger.info(f"Epoch {epoch}: {converged_count}/{self.num_nodes} nodes converged "
                        f"({feedback_info['convergence_rate']:.2%})")
        
        return feedback_info
    
    def get_adaptive_batch(self, batch_size: int, epoch: int, 
                          train_mask: torch.Tensor) -> torch.Tensor:
        """获取自适应生成的batch"""
        sampled_nodes = self.adaptive_sampler.sample_nodes(batch_size, epoch, train_mask)
        self.feedback_stats['sampling_adaptations'] += 1
        
        return sampled_nodes
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息"""
        converged_nodes = self.convergence_tracker.get_converged_nodes()
        
        return {
            'feedback_stats': self.feedback_stats.copy(),
            'converged_nodes_count': len(converged_nodes),
            'convergence_rate': len(converged_nodes) / self.num_nodes,
            'sampling_strategy': self.adaptive_sampler.sampling_strategy,
            'exploration_rate': self.adaptive_sampler.exploration_rate
        }
    
    def adjust_sampling_strategy(self, epoch: int, convergence_rate: float):
        """根据收敛率调整采样策略"""
        if convergence_rate > 0.8:
            # 大部分节点收敛，增加探索率
            self.adaptive_sampler.exploration_rate = min(0.3, 
                self.adaptive_sampler.exploration_rate * 1.1)
        elif convergence_rate < 0.3:
            # 很少节点收敛，减少探索率，专注训练
            self.adaptive_sampler.exploration_rate = max(0.05, 
                self.adaptive_sampler.exploration_rate * 0.9)
        
        self.logger.info(f"Epoch {epoch}: Adjusted exploration rate to "
                        f"{self.adaptive_sampler.exploration_rate:.3f}")


class InfoFeedbackSystem:
    """信息互馈系统主类"""
    
    def __init__(self, num_nodes: int, device: torch.device, 
                 enable_feedback: bool = True):
        self.device = device
        self.enable_feedback = enable_feedback
        
        if enable_feedback:
            self.feedback_controller = FeedbackController(num_nodes, device)
            self.logger = logging.getLogger(__name__)
            self.logger.info("信息互馈系统已启用")
        else:
            self.feedback_controller = None
            self.logger = logging.getLogger(__name__)
            self.logger.info("信息互馈系统已禁用，使用标准训练流程")
    
    def is_enabled(self) -> bool:
        """检查系统是否启用"""
        return self.enable_feedback and self.feedback_controller is not None
    
    def process_feedback(self, node_ids: torch.Tensor, losses: torch.Tensor,
                        gradients: torch.Tensor, embeddings: torch.Tensor, 
                        epoch: int) -> Optional[Dict]:
        """处理训练反馈"""
        if not self.is_enabled():
            return None
            
        return self.feedback_controller.process_training_feedback(
            node_ids, losses, gradients, embeddings, epoch)
    
    def get_adaptive_batch(self, batch_size: int, epoch: int, 
                          train_mask: torch.Tensor) -> torch.Tensor:
        """获取自适应batch"""
        if not self.is_enabled():
            # 如果系统未启用，返回随机采样
            train_nodes = torch.where(train_mask)[0]
            if len(train_nodes) == 0:
                return torch.tensor([], dtype=torch.long, device=train_mask.device)
            elif len(train_nodes) <= batch_size:
                return train_nodes
            else:
                indices = torch.randperm(len(train_nodes))[:batch_size]
                return train_nodes[indices]
        
        return self.feedback_controller.get_adaptive_batch(batch_size, epoch, train_mask)
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        if not self.is_enabled():
            return {'status': 'disabled'}
        
        return self.feedback_controller.get_system_status()
    
    def adjust_strategy(self, epoch: int, convergence_rate: float):
        """调整系统策略"""
        if self.is_enabled():
            self.feedback_controller.adjust_sampling_strategy(epoch, convergence_rate)
