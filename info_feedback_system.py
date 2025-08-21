import torch
import numpy as np
from typing import Dict, Optional
from collections import defaultdict, deque
import logging
from torch.jit import script

# 安全 isin，兼容旧版本 PyTorch
def safe_isin(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """安全的isin操作，兼容旧版本PyTorch"""
    try:
        return torch.isin(a, b)
    except AttributeError:
        if a.numel() == 0 or b.numel() == 0:
            return torch.zeros_like(a, dtype=torch.bool)
        a_cpu = a.detach().cpu().numpy()
        b_cpu = b.detach().cpu().numpy()
        mask_np = np.isin(a_cpu, b_cpu)
        return torch.from_numpy(mask_np).to(a.device)

@script
def safe_isin_jit(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """JIT优化版本的isin，假设torch.isin可用"""
    return torch.isin(a, b)

@script
def compute_cosine_similarity_batch(embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
    """JIT优化的批量余弦相似性计算"""
    # 归一化向量 - 修复JIT兼容性
    embed1_norm = embed1 / (torch.norm(embed1, p=2, dim=1, keepdim=True) + 1e-8)
    embed2_norm = embed2 / (torch.norm(embed2, p=2, dim=1, keepdim=True) + 1e-8)
    # 计算余弦相似性
    return torch.sum(embed1_norm * embed2_norm, dim=1)

@script
def batch_ema_update(ema_embeddings: torch.Tensor, new_embeddings: torch.Tensor, 
                    decay: float) -> torch.Tensor:
    """批量EMA更新"""
    return decay * ema_embeddings + (1.0 - decay) * new_embeddings

@script  
def filter_valid_nodes(node_ids: torch.Tensor, max_nodes: int) -> torch.Tensor:
    """过滤有效节点ID"""
    return node_ids[node_ids < max_nodes]

class NodeConvergenceTracker:
    """节点收敛状态跟踪器 - 优化版本"""
    
    def __init__(self, num_nodes: int, convergence_threshold: float = 0.01, 
                 patience: int = 2, min_epochs: int = 1, embedding_dim: int = 64,
                 similarity_threshold: float = 0.80, ema_decay: float = 0.9, device: torch.device = None,
                 use_simple_convergence: bool = True):
        self.num_nodes = num_nodes
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.min_epochs = min_epochs
        self.embedding_dim = embedding_dim
        
        # 核心参数：嵌入相似性阈值
        self.similarity_threshold = similarity_threshold  # 嵌入相似性阈值
        self.ema_decay = ema_decay  # EMA衰减率
        
        # 设备管理
        self.device = device or torch.device('cpu')
        
        # 性能优化：使用简单收敛判断
        self.use_simple_convergence = use_simple_convergence
        
        # 内存优化：使用连续内存布局
        if use_simple_convergence:
            # 简单版本：只存储最近两次嵌入
            self.node_last_embeddings = torch.zeros(num_nodes, embedding_dim, dtype=torch.float32, 
                                                   device=self.device).contiguous()
            self.node_prev_embeddings = torch.zeros(num_nodes, embedding_dim, dtype=torch.float32, 
                                                   device=self.device).contiguous()
            self.node_update_count = torch.zeros(num_nodes, dtype=torch.long, 
                                                device=self.device).contiguous()
        else:
            # 完整EMA版本
            self.node_ema_embeddings = torch.zeros(num_nodes, embedding_dim, dtype=torch.float32, 
                                                  device=self.device).contiguous()
            self.node_ema_count = torch.zeros(num_nodes, dtype=torch.long, 
                                             device=self.device).contiguous()
            self.node_prev_ema = torch.zeros(num_nodes, embedding_dim, dtype=torch.float32, 
                                            device=self.device).contiguous()
        
        # 历史缓存（保留用于兼容性，但主要使用EMA）
        self.node_loss_history = defaultdict(lambda: deque(maxlen=patience))
        self.node_grad_history = defaultdict(lambda: deque(maxlen=patience))
        
        # 状态集合
        self.converged_nodes = set()
        self.convergence_epochs = defaultdict(int)
        self.eligible_nodes = set()  # 历史长度>=patience 的节点
        
        # 节点重要性分数
        self.node_importance_scores = torch.ones(num_nodes, dtype=torch.float32, device=self.device)
        
        # 缓存：存储上一轮的EMA嵌入用于相似性计算
        self.node_prev_ema = torch.zeros(num_nodes, embedding_dim, dtype=torch.float32, device=self.device)
    
    def update_embedding_dim(self, new_dim: int):
        """更新嵌入维度"""
        if new_dim != self.embedding_dim:
            self.embedding_dim = new_dim
            # 重新初始化张量
            if self.use_simple_convergence:
                self.node_last_embeddings = torch.zeros(self.num_nodes, new_dim, dtype=torch.float32, device=self.device)
                self.node_prev_embeddings = torch.zeros(self.num_nodes, new_dim, dtype=torch.float32, device=self.device)
            else:
                self.node_ema_embeddings = torch.zeros(self.num_nodes, new_dim, dtype=torch.float32, device=self.device)
                self.node_prev_ema = torch.zeros(self.num_nodes, new_dim, dtype=torch.float32, device=self.device)
    
    def _compute_embedding_similarity_batch(self, embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """批量计算嵌入向量的相似性（余弦相似性）- JIT优化版本"""
        return compute_cosine_similarity_batch(embed1, embed2)
    
    def _compute_embedding_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        """计算两个嵌入向量的相似性（余弦相似性）- 单向量版本"""
        # 归一化向量
        embed1_norm = embed1 / (embed1.norm() + 1e-8)
        embed2_norm = embed2 / (embed2.norm() + 1e-8)
        
        # 计算余弦相似性
        similarity = torch.dot(embed1_norm, embed2_norm).item()
        return similarity
    
    def _check_convergence_batch(self, node_ids: torch.Tensor, epoch: int) -> torch.Tensor:
        """批量检查节点是否收敛 - 基于EMA相似性"""
        if epoch < self.min_epochs:
            return torch.zeros(len(node_ids), dtype=torch.bool, device=node_ids.device)
        
        # 获取当前EMA和上一轮EMA
        current_ema = self.node_ema_embeddings[node_ids]
        prev_ema = self.node_prev_ema[node_ids]
        
        # 计算相似性
        similarities = self._compute_embedding_similarity_batch(current_ema, prev_ema)
        
        # 检查是否达到阈值且EMA更新次数足够
        ema_counts = self.node_ema_count[node_ids]
        convergence_mask = (similarities >= self.similarity_threshold) & (ema_counts >= self.patience)
        
        return convergence_mask
    
    def _check_convergence(self, node_id: int, epoch: int) -> bool:
        """检查单个节点是否收敛 - 基于相似性"""
        if epoch < self.min_epochs:
            return False
            
        if node_id in self.converged_nodes:
            return False
        
        if self.use_simple_convergence:
            # 简单版本：检查更新次数和相似性
            if self.node_update_count[node_id] < self.patience:
                return False
            
            # 计算当前嵌入与上一轮嵌入的相似性
            current_embedding = self.node_last_embeddings[node_id]
            prev_embedding = self.node_prev_embeddings[node_id]
            
            # 检查张量是否有效
            if current_embedding.numel() == 0 or prev_embedding.numel() == 0:
                return False
            
            similarity = self._compute_embedding_similarity(current_embedding, prev_embedding)
            return similarity >= self.similarity_threshold
        else:
            # 完整EMA版本
            if self.node_ema_count[node_id] < self.patience:
                return False
            
            # 计算当前EMA与上一轮EMA的相似性
            current_ema = self.node_ema_embeddings[node_id]
            prev_ema = self.node_prev_ema[node_id]
            
            similarity = self._compute_embedding_similarity(current_ema, prev_ema)
            return similarity >= self.similarity_threshold
    
    def update_node_info(self, node_ids: torch.Tensor, losses: torch.Tensor, 
                        gradients: torch.Tensor, embeddings: torch.Tensor, epoch: int):
        """更新节点的训练信息 - 批量优化版本"""
        if node_ids.numel() == 0:
            return
            
        # 动态更新嵌入维度
        if (embeddings is not None and 
            embeddings.numel() > 0 and
            embeddings.dim() > 1 and
            embeddings.size(1) != self.embedding_dim):
            self.update_embedding_dim(embeddings.size(1))
        
        # 批量更新嵌入
        if embeddings is not None and embeddings.numel() > 0:
            # 确保所有张量在同一设备上
            if node_ids.device != self.device:
                node_ids = node_ids.to(self.device)
            if embeddings.device != self.device:
                embeddings = embeddings.to(self.device)
            
            # 过滤有效节点ID
            valid_mask = node_ids < self.num_nodes
            if not valid_mask.any():
                return
                
            valid_node_ids = node_ids[valid_mask]
            valid_embeddings = embeddings[valid_mask]
            valid_losses = losses[valid_mask] if losses is not None else None
            
            # 批量更新嵌入
            self._batch_update_embeddings(valid_node_ids, valid_embeddings, valid_losses, epoch)
    
    def _batch_update_embeddings(self, node_ids: torch.Tensor, embeddings: torch.Tensor, 
                                losses: torch.Tensor, epoch: int):
        """批量更新节点嵌入 - 向量化实现"""
        if self.use_simple_convergence:
            # 批量更新简单版本 - 处理初始化情况
            # 对于第一次更新的节点，不保存全零向量到prev
            first_update_mask = self.node_update_count[node_ids] == 0
            
            if first_update_mask.any():
                # 第一次更新：直接设置current，prev保持为零（不参与收敛检查）
                first_nodes = node_ids[first_update_mask]
                self.node_last_embeddings[first_nodes] = embeddings[first_update_mask].detach()
            
            if (~first_update_mask).any():
                # 非第一次更新：保存旧的current到prev，然后更新current
                update_nodes = node_ids[~first_update_mask]
                update_embeds = embeddings[~first_update_mask].detach()
                self.node_prev_embeddings[update_nodes] = self.node_last_embeddings[update_nodes].clone()
                self.node_last_embeddings[update_nodes] = update_embeds
            
            self.node_update_count[node_ids] = self.node_update_count[node_ids] + 1
            
            # 批量检查收敛
            eligible_mask = self.node_update_count[node_ids] >= self.patience
            if eligible_mask.any():
                eligible_nodes = node_ids[eligible_mask]
                # 添加到eligible集合
                for node_id in eligible_nodes:
                    self.eligible_nodes.add(int(node_id.item()))
                
                # 批量收敛检查
                if epoch >= self.min_epochs:
                    current_embeds = self.node_last_embeddings[eligible_nodes]
                    prev_embeds = self.node_prev_embeddings[eligible_nodes]
                    
                    similarities = self._compute_embedding_similarity_batch(current_embeds, prev_embeds)
                    converged_mask = similarities >= self.similarity_threshold
                    
                    
                    if converged_mask.any():
                        converged_nodes = eligible_nodes[converged_mask]
                        for node_id in converged_nodes:
                            node_id_int = int(node_id.item())
                            self.converged_nodes.add(node_id_int)
                            self.convergence_epochs[node_id_int] = epoch
        else:
            # 批量更新EMA版本 - 内存优化
            self.node_prev_ema[node_ids].copy_(self.node_ema_embeddings[node_ids])
            
            # 区分初始化和更新
            first_update_mask = self.node_ema_count[node_ids] == 0
            update_mask = ~first_update_mask
            
            if first_update_mask.any():
                first_nodes = node_ids[first_update_mask]
                first_embeds = embeddings[first_update_mask].detach()
                self.node_ema_embeddings[first_nodes].copy_(first_embeds)
            
            if update_mask.any():
                update_nodes = node_ids[update_mask]
                update_embeds = embeddings[update_mask].detach()
                # 使用JIT优化的EMA更新
                self.node_ema_embeddings[update_nodes].copy_(
                    batch_ema_update(self.node_ema_embeddings[update_nodes], update_embeds, self.ema_decay)
                )
            
            self.node_ema_count[node_ids].add_(1)
            
            # 批量检查收敛
            eligible_mask = self.node_ema_count[node_ids] >= self.patience
            if eligible_mask.any():
                eligible_nodes = node_ids[eligible_mask]
                for node_id in eligible_nodes:
                    self.eligible_nodes.add(int(node_id.item()))
                
                if epoch >= self.min_epochs:
                    current_ema = self.node_ema_embeddings[eligible_nodes]
                    prev_ema = self.node_prev_ema[eligible_nodes]
                    
                    similarities = self._compute_embedding_similarity_batch(current_ema, prev_ema)
                    converged_mask = similarities >= self.similarity_threshold
                    
                    if converged_mask.any():
                        converged_nodes = eligible_nodes[converged_mask]
                        for node_id in converged_nodes:
                            node_id_int = int(node_id.item())
                            self.converged_nodes.add(node_id_int)
                            self.convergence_epochs[node_id_int] = epoch
        
        # 批量更新历史记录（简化版本）- 减少循环开销
        if losses is not None:
            # 预先转换为CPU numpy避免重复转换
            losses_cpu = losses.detach().cpu().numpy()
            node_ids_cpu = node_ids.detach().cpu().numpy()
            
            for i, node_id_int in enumerate(node_ids_cpu):
                loss_val = float(losses_cpu[i])
                self.node_loss_history[node_id_int].append(loss_val)
                self.node_grad_history[node_id_int].append(loss_val)
    
    def get_eligible_count(self) -> int:
        """获取历史长度>=patience的节点数量"""
        return len(self.eligible_nodes)

    def get_converged_nodes(self) -> set:
        """获取已收敛的节点"""
        return self.converged_nodes.copy()
    
    def get_node_importance(self, node_ids: torch.Tensor) -> torch.Tensor:
        """获取节点的重要性分数（兼容CUDA索引）"""
        if isinstance(node_ids, torch.Tensor):
            # 确保索引在正确的设备上
            if node_ids.device != self.device:
                node_ids = node_ids.to(self.device)
            scores = self.node_importance_scores[node_ids]
            return scores
        else:
            # 兼容列表/ndarray
            node_ids_tensor = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)
            scores = self.node_importance_scores[node_ids_tensor]
            return scores
    
    def update_importance_scores(self, node_ids: torch.Tensor, 
                               new_scores: torch.Tensor):
        """更新节点重要性分数"""
        # 确保索引在正确的设备上
        if node_ids.device != self.device:
            node_ids = node_ids.to(self.device)
        if new_scores.device != self.device:
            new_scores = new_scores.to(self.device)
        self.node_importance_scores[node_ids] = new_scores
    
    def get_convergence_stats(self) -> dict:
        """获取收敛统计信息"""
        total_nodes = self.num_nodes
        converged_count = len(self.converged_nodes)
        
        # 计算平均嵌入稳定性
        avg_stability = 0.0
        if converged_count > 0:
            stabilities = []
            for node_id in self.converged_nodes:
                # 安全地获取EMA嵌入
                if node_id < self.num_nodes:
                    current_ema = self.node_ema_embeddings[node_id]
                    prev_ema = self.node_prev_ema[node_id]
                    # 检查张量是否有效
                    if current_ema.numel() > 0 and prev_ema.numel() > 0:
                        stability = self._compute_embedding_similarity(current_ema, prev_ema)
                        stabilities.append(stability)
            
            if stabilities:
                avg_stability = np.mean(stabilities)
        
        return {
            'total_nodes': total_nodes,
            'converged_nodes': converged_count,
            'convergence_rate': converged_count / total_nodes,
            'avg_embedding_stability': avg_stability,
            'similarity_threshold': self.similarity_threshold
        }


class AdaptiveSampler:
    """自适应采样器，根据训练反馈调整采样策略"""
    
    def __init__(self, num_nodes: int, convergence_tracker: NodeConvergenceTracker,
                 sampling_strategy: str = 'adaptive_importance',
                 candidate_pool_size: Optional[int] = None,
                 keep_sampling_stats: bool = False):
        self.num_nodes = num_nodes
        self.convergence_tracker = convergence_tracker
        # 规范化策略名：no_importance 视为 random（均匀）
        self.sampling_strategy = sampling_strategy
        self.candidate_pool_size = candidate_pool_size
        
        self.keep_sampling_stats = keep_sampling_stats
        if keep_sampling_stats:
            self.sampling_counts = torch.zeros(num_nodes, dtype=torch.long)
            self.last_sampling_epoch = torch.full((num_nodes,), -1, dtype=torch.long)
        else:
            self.sampling_counts = None
            self.last_sampling_epoch = None
        
        self.exploration_rate = 0.1
        self.convergence_penalty = 0.5
    
    def _filter_converged(self, nodes: torch.Tensor) -> torch.Tensor:
        """从节点集合中过滤掉已收敛节点"""
        converged = list(self.convergence_tracker.get_converged_nodes())
        if len(converged) == 0 or nodes.numel() == 0:
            return nodes
        conv = torch.tensor(converged, dtype=nodes.dtype, device=nodes.device)
        keep_mask = ~safe_isin(nodes, conv)
        return nodes[keep_mask]
    
    def _maybe_record_stats(self, nodes: torch.Tensor, epoch: int):
        if self.keep_sampling_stats and nodes.numel() > 0:
            nodes_cpu = nodes.detach().to('cpu')
            self.sampling_counts[nodes_cpu] += 1
            self.last_sampling_epoch[nodes_cpu] = epoch
        
    def _pick_candidates(self, train_nodes: torch.Tensor) -> torch.Tensor:
        if self.candidate_pool_size is None or len(train_nodes) <= self.candidate_pool_size:
            return train_nodes
        idx = torch.randperm(len(train_nodes), device=train_nodes.device)[:self.candidate_pool_size]
        return train_nodes[idx]
    
    def sample_nodes(self, batch_size: int, epoch: int, 
                    train_mask: torch.Tensor) -> torch.Tensor:
        # no_importance == random（均匀）
        if self.sampling_strategy in ('no_importance', 'random'):
            return self._random_sampling(batch_size, train_mask)
        elif self.sampling_strategy == 'convergence_aware':
            return self._convergence_aware_sampling(batch_size, epoch, train_mask)
        else:
            return self._adaptive_importance_sampling(batch_size, epoch, train_mask)
    
    def _adaptive_importance_sampling(self, batch_size: int, epoch: int, 
                                    train_mask: torch.Tensor) -> torch.Tensor:
        train_nodes = torch.where(train_mask)[0]
        # 排除已收敛节点
        train_nodes = self._filter_converged(train_nodes)
        if len(train_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=train_mask.device)
        if len(train_nodes) <= batch_size:
            self._maybe_record_stats(train_nodes, epoch)
            return train_nodes
        # 正确获取候选集合与其重要性分数
        candidates = self._pick_candidates(train_nodes)
        importance_scores = self.convergence_tracker.get_node_importance(candidates).clone().to(candidates.device)
        
        # 不再惩罚收敛节点（已过滤）；保留探索扰动
        if self.exploration_rate > 0:
            exploration_mask = (torch.rand(len(candidates), device=candidates.device) < self.exploration_rate)
            if exploration_mask.any():
                importance_scores = torch.where(exploration_mask, torch.ones_like(importance_scores, device=candidates.device), importance_scores)
        
        total = importance_scores.sum()
        if total <= 0:
            idx = torch.randperm(len(candidates), device=candidates.device)[:batch_size]
            sampled_nodes = candidates[idx]
        else:
            probs = importance_scores / total
            sampled_idx = torch.multinomial(probs, batch_size, replacement=False)
            sampled_nodes = candidates[sampled_idx]
        self._maybe_record_stats(sampled_nodes, epoch)
        return sampled_nodes
    
    def _convergence_aware_sampling(self, batch_size: int, epoch: int,
                                   train_mask: torch.Tensor) -> torch.Tensor:
        train_nodes = torch.where(train_mask)[0]
        # 排除已收敛节点
        train_nodes = self._filter_converged(train_nodes)
        if len(train_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=train_mask.device)
        candidates = self._pick_candidates(train_nodes)
        
        # 这里 candidates 已经不包含收敛节点，直接截断采样
        if len(candidates) >= batch_size:
            sampled_nodes = candidates[:batch_size]
        else:
            sampled_nodes = candidates
        
        self._maybe_record_stats(sampled_nodes, epoch)
        return sampled_nodes
    
    def _random_sampling(self, batch_size: int, train_mask: torch.Tensor) -> torch.Tensor:
        train_nodes = torch.where(train_mask)[0]
        # 排除已收敛节点
        train_nodes = self._filter_converged(train_nodes)
        if len(train_nodes) == 0:
            return torch.tensor([], dtype=torch.long, device=train_mask.device)
        if len(train_nodes) <= batch_size:
            return train_nodes
        else:
            indices = torch.randperm(len(train_nodes), device=train_nodes.device)[:batch_size]
            return train_nodes[indices]


class FeedbackController:
    """反馈控制器，协调整个信息互馈系统 - 优化版本"""
    
    def __init__(self, num_nodes: int, device: torch.device,
                 sampler_candidate_pool_size: Optional[int] = None,
                 keep_sampling_stats: bool = False,
                 similarity_threshold: Optional[float] = 0.80,
                 patience: Optional[int] = 2,
                 min_epochs: Optional[int] = 1,
                 sampling_strategy: Optional[str] = None,
                 feedback_batch_cap: Optional[int] = 1000,
                 use_simple_convergence: bool = True):
        self.device = device
        self.num_nodes = num_nodes
        
        # 新增：反馈批大小上限，用于限制反馈处理的节点数量
        self.feedback_batch_cap = feedback_batch_cap
        
        tracker_args = {}
        if similarity_threshold is not None:
            tracker_args['similarity_threshold'] = similarity_threshold
        if patience is not None:
            tracker_args['patience'] = patience
        if min_epochs is not None:
            tracker_args['min_epochs'] = min_epochs
        # 传递设备参数和收敛模式
        tracker_args['device'] = device
        tracker_args['use_simple_convergence'] = use_simple_convergence
        self.convergence_tracker = NodeConvergenceTracker(num_nodes, **tracker_args)
        self.adaptive_sampler = AdaptiveSampler(
            num_nodes, self.convergence_tracker,
            sampling_strategy=(sampling_strategy or 'adaptive_importance'),
            candidate_pool_size=sampler_candidate_pool_size,
            keep_sampling_stats=keep_sampling_stats
        )
        
        self.feedback_stats = {
            'total_feedback_cycles': 0,
            'nodes_converged': 0,
            'sampling_adaptations': 0,
            'feedback_batch_capped': 0  # 新增：记录被截断的反馈次数
        }
        self.logger = logging.getLogger(__name__)
        
    def process_training_feedback(self, node_ids: torch.Tensor, losses: torch.Tensor,
                                gradients: torch.Tensor, embeddings: torch.Tensor, 
                                epoch: int) -> Dict:
        """处理训练反馈信息 - 优化版本"""
        # 边界情况处理：如果输入为空，返回默认反馈信息
        if node_ids.numel() == 0:
            feedback_info = {
                'epoch': epoch,
                'converged_nodes_count': 0,
                'convergence_rate': 0.0,
                'eligible_convergence_rate': 0.0,
                'eligible_nodes': 0,
                'total_nodes': self.num_nodes,
                'feedback_batch_size': 0,
                'feedback_batch_capped': False
            }
            self.logger.info(f"Epoch {epoch}: Empty batch, no feedback to process")
            return feedback_info
        
        # 优化：应用反馈批大小上限
        original_batch_size = len(node_ids)
        if self.feedback_batch_cap is not None and original_batch_size > self.feedback_batch_cap:
            # 随机选择子集进行反馈处理
            indices = torch.randperm(original_batch_size, device=node_ids.device)[:self.feedback_batch_cap]
            node_ids = node_ids[indices]
            losses = losses[indices]
            if gradients is not None:
                gradients = gradients[indices]
            if embeddings is not None:
                embeddings = embeddings[indices]
            
            self.feedback_stats['feedback_batch_capped'] += 1
            self.logger.info(f"Epoch {epoch}: Feedback batch capped from {original_batch_size} to {len(node_ids)} nodes")
        
        # 使用torch.no_grad()优化反馈处理
        with torch.no_grad():
            # 更新节点收敛状态
            self.convergence_tracker.update_node_info(node_ids, losses, gradients, 
                                                    embeddings, epoch)
        
        # 计算反馈统计
        converged_count = len(self.convergence_tracker.get_converged_nodes())
        eligible_count = self.convergence_tracker.get_eligible_count()
        eligible_rate = (converged_count / eligible_count) if eligible_count > 0 else 0.0
        global_rate = converged_count / self.num_nodes
        
        feedback_info = {
            'epoch': epoch,
            'converged_nodes_count': converged_count,
            'convergence_rate': global_rate,
            'eligible_convergence_rate': eligible_rate,
            'eligible_nodes': eligible_count,
            'total_nodes': self.num_nodes,
            'feedback_batch_size': len(node_ids),
            'feedback_batch_capped': original_batch_size > len(node_ids) if self.feedback_batch_cap else False
        }
        
        # 记录反馈统计
        self.feedback_stats['total_feedback_cycles'] += 1
        self.feedback_stats['nodes_converged'] = converged_count
        
        self.logger.info(
            f"Epoch {epoch}: {converged_count}/{self.num_nodes} nodes converged "
            f"(global {global_rate:.2%}, eligible {eligible_rate:.2%} over {eligible_count} nodes) "
            f"[feedback: {len(node_ids)}/{original_batch_size}]"
        )
        
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
        eligible_count = self.convergence_tracker.get_eligible_count()
        eligible_rate = (len(converged_nodes) / eligible_count) if eligible_count > 0 else 0.0
        return {
            'feedback_stats': self.feedback_stats.copy(),
            'converged_nodes_count': len(converged_nodes),
            'convergence_rate': len(converged_nodes) / self.num_nodes,
            'eligible_convergence_rate': eligible_rate,
            'eligible_nodes': eligible_count,
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


class EarlyStopping:
    """基于收敛率的早停机制"""
    
    def __init__(self, threshold: float = 0.02, patience: int = 3, 
                 min_epochs: int = 5, enabled: bool = False):
        self.threshold = threshold  # 收敛率阈值
        self.patience = patience    # 耐心值
        self.min_epochs = min_epochs # 最小训练轮数
        self.enabled = enabled
        
        self.consecutive_count = 0  # 连续满足条件的epoch数
        self.convergence_history = []  # 收敛率历史
        self.should_stop = False
        self.stop_reason = ""
        
        self.logger = logging.getLogger(__name__)
        
    def check(self, epoch: int, convergence_rate: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            epoch: 当前epoch
            convergence_rate: 当前收敛率
            
        Returns:
            bool: 是否应该停止训练
        """
        if not self.enabled:
            return False
        
        self.convergence_history.append(convergence_rate)
        
        # 检查最小epoch要求
        if epoch < self.min_epochs:
            self.logger.info(f"Early Stopping: Epoch {epoch} < min_epochs {self.min_epochs}, continue training")
            return False
        
        # 检查收敛率阈值
        if convergence_rate >= self.threshold:
            self.consecutive_count += 1
            self.logger.info(
                f"Early Stopping: Convergence rate {convergence_rate:.3%} >= threshold {self.threshold:.1%} "
                f"({self.consecutive_count}/{self.patience})"
            )
            
            if self.consecutive_count >= self.patience:
                self.should_stop = True
                self.stop_reason = (f"Convergence rate {convergence_rate:.3%} >= {self.threshold:.1%} "
                                  f"for {self.patience} consecutive epochs")
                self.logger.info(f"Early Stopping: TRIGGERED - {self.stop_reason}")
                return True
        else:
            if self.consecutive_count > 0:
                self.logger.info(
                    f"Early Stopping: Convergence rate {convergence_rate:.3%} < threshold {self.threshold:.1%}, "
                    f"resetting counter (was {self.consecutive_count})"
                )
            self.consecutive_count = 0
        
        return False
    
    def get_status(self) -> Dict:
        """获取早停状态"""
        return {
            'enabled': self.enabled,
            'threshold': self.threshold,
            'patience': self.patience,
            'min_epochs': self.min_epochs,
            'consecutive_count': self.consecutive_count,
            'convergence_history': self.convergence_history[-10:],  # 最近10个epoch
            'should_stop': self.should_stop,
            'stop_reason': self.stop_reason
        }


class InfoFeedbackSystem:
    """信息互馈系统主类 - 优化版本"""
    
    def __init__(self, num_nodes: int, device: torch.device, 
                 enable_feedback: bool = True,
                 sampler_candidate_pool_size: Optional[int] = None,
                 keep_sampling_stats: bool = False,
                 similarity_threshold: Optional[float] = 0.80,
                 patience: Optional[int] = 2,
                 min_epochs: Optional[int] = 1,
                 sampling_strategy: Optional[str] = None,
                 feedback_batch_cap: Optional[int] = 1000,
                 use_simple_convergence: bool = True,
                 # 早停机制参数
                 enable_early_stopping: bool = False,
                 early_stop_threshold: float = 0.02,
                 early_stop_patience: int = 3,
                 min_epochs_before_stop: int = 5):
        self.device = device
        self.enable_feedback = enable_feedback
        
        # 初始化早停机制
        self.early_stopping = EarlyStopping(
            threshold=early_stop_threshold,
            patience=early_stop_patience,
            min_epochs=min_epochs_before_stop,
            enabled=enable_early_stopping
        )
        
        if enable_feedback:
            self.feedback_controller = FeedbackController(
                num_nodes, device,
                sampler_candidate_pool_size=sampler_candidate_pool_size,
                keep_sampling_stats=keep_sampling_stats,
                similarity_threshold=similarity_threshold,
                patience=patience,
                min_epochs=min_epochs,
                sampling_strategy=sampling_strategy,
                feedback_batch_cap=feedback_batch_cap,
                use_simple_convergence=use_simple_convergence
            )
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"信息互馈系统已启用 [feedback_batch_cap: {feedback_batch_cap}, simple_convergence: {use_simple_convergence}]")
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
                indices = torch.randperm(len(train_nodes), device=train_nodes.device)[:batch_size]
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
    
    def check_early_stopping(self, epoch: int, convergence_rate: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            epoch: 当前epoch
            convergence_rate: 当前收敛率
            
        Returns:
            bool: 是否应该停止训练
        """
        return self.early_stopping.check(epoch, convergence_rate)
    
    def should_early_stop(self, epoch: int) -> tuple[bool, str]:
        """
        检查是否应该早停
        
        Args:
            epoch: 当前epoch
            
        Returns:
            tuple: (是否应该停止, 停止原因)
        """
        if not self.early_stopping.enabled:
            return False, ""
        
        # 获取当前收敛率
        if self.feedback_controller is None:
            return False, ""
            
        converged_count = len(self.feedback_controller.convergence_tracker.get_converged_nodes())
        convergence_rate = converged_count / self.feedback_controller.num_nodes
        
        # 检查早停条件
        should_stop = self.early_stopping.check(epoch, convergence_rate)
        stop_reason = self.early_stopping.stop_reason if should_stop else ""
        
        return should_stop, stop_reason
    
    def get_early_stopping_status(self) -> Dict:
        """获取早停状态"""
        return self.early_stopping.get_status()
