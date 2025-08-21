from coo_graph import Parted_COO_Graph
from coo_graph import Full_COO_Graph
from coo_graph import Full_COO_Graph_Large
from coo_graph import Full_COO_Graph_CPU
from models import GCN, GAT, CachedGCN, DecoupleGCN, TensplitGCN, TensplitGCNLARGE, TensplitGCNCPU, TensplitGAT
from info_feedback_system import InfoFeedbackSystem

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
from sklearn.metrics import f1_score
from dist_utils import DistEnv
import logging
import torch.distributed as dist

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def f1(y_true, y_pred, multilabel=True):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    if multilabel:
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0
        for node in [10,100,1000]:
            DistEnv.env.logger.log('pred', y_pred[node] , rank=0)
            DistEnv.env.logger.log('true', y_true[node] , rank=0)
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
           f1_score(y_true, y_pred, average="macro")

def train_with_feedback(g, env, args, feedback_system: InfoFeedbackSystem):
    """集成信息互馈的训练函数（性能优化版）"""
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        loss_func = nn.CrossEntropyLoss(reduction='none')
    else:
        loss_func = nn.BCEWithLogitsLoss(reduction='none')

    # 训练前快速统计：本 rank 训练节点总数（用于验证上限）
    try:
        if g.local_labels[g.local_train_mask].size(0) > 0:
            _train_nodes_static = torch.where(g.local_train_mask)[0]
            env.logger.log(f"Local train nodes (rank {env.rank}): {_train_nodes_static.numel()}", rank=0)
    except Exception:
        pass

    # 性能优化：动态调整反馈频率
    base_feedback_every = getattr(args, 'feedback_every', 5)
    adaptive_feedback = getattr(args, 'adaptive_feedback', True)
    
    def get_feedback_interval(epoch: int, total_epochs: int) -> int:
        """动态计算反馈间隔"""
        if not adaptive_feedback:
            return base_feedback_every
        
        # 早期阶段：每个epoch都反馈，快速建立收敛基线
        if epoch < total_epochs * 0.4:
            return 1
        # 中期阶段：适中反馈
        elif epoch < total_epochs * 0.8:
            return max(2, base_feedback_every // 2)  
        # 后期阶段：减少反馈频率
        else:
            return base_feedback_every

    for epoch in range(args.epoch):
        with env.timer.timing('epoch'):
            with autocast(env.half_enabled):
                with env.timer.timing('forward'):
                    outputs = model(g.features)
                optimizer.zero_grad()

                if g.local_labels[g.local_train_mask].size(0) > 0:
                    train_nodes = torch.where(g.local_train_mask)[0]

                    # 动态调整反馈频率
                    current_feedback_every = get_feedback_interval(epoch, args.epoch)
                    do_feedback = feedback_system.is_enabled() and (epoch % current_feedback_every == 0)
                    
                    if do_feedback:
                        # 性能优化：自适应batch大小
                        base_batch_size = getattr(args, 'batch_size', len(train_nodes))
                        adaptive_batch_size = min(base_batch_size, len(train_nodes))
                        
                        # 根据节点收敛率调整batch大小
                        if epoch > 0 and epoch % 10 == 0:
                            status = feedback_system.get_status()
                            if 'convergence_rate' in status and status['convergence_rate'] > 0.5:
                                # 大量节点已收敛，减小batch大小
                                adaptive_batch_size = max(adaptive_batch_size // 2, 100)
                        
                        with env.timer.timing('feedback.sample'):
                            batch_nodes = feedback_system.get_adaptive_batch(
                                adaptive_batch_size, epoch, g.local_train_mask)
                        
                        # 如果所有节点都已收敛，直接早停
                        if len(batch_nodes) == 0:
                            env.logger.log(f"Epoch {epoch} | All local training nodes converged, triggering early stop", rank=0)
                            break  # 直接跳出训练循环
                        else:
                            # 正常训练逻辑
                            with env.timer.timing('loss'):
                                if g.labels.dim() == 1:
                                    batch_losses = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes])
                                else:
                                    batch_losses = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes]).mean(dim=1)
                                loss = batch_losses.mean()

                            with env.timer.timing_cuda('backward'):
                                loss.backward()

                            # 在完成 backward 后，再做轻量的互馈处理（不涉及再反向）
                            if do_feedback:
                                # 优化：使用torch.no_grad()减少内存开销
                                with torch.no_grad():
                                    node_gradients = batch_losses.detach()
                                    batch_outputs = outputs[batch_nodes].detach()
                                    
                                with env.timer.timing('feedback.process'):
                                    feedback_info = feedback_system.process_feedback(
                                        batch_nodes, batch_losses.detach(), node_gradients, batch_outputs, epoch)
                                
                                if feedback_info:
                                    env.logger.log(f"Epoch {epoch} | Feedback: {feedback_info['convergence_rate']:.3f} nodes converged (interval: {current_feedback_every})", rank=0)

                                # 快速验证：本 rank 训练集内已收敛与剩余数量 + 全局汇总
                                try:
                                    train_nodes_cpu_set = set(map(int, train_nodes.detach().to('cpu').numpy().tolist()))
                                    converged_set = set()
                                    # 读取当前已收敛节点集合
                                    if hasattr(feedback_system, 'feedback_controller') and feedback_system.feedback_controller is not None:
                                        converged_set = set(feedback_system.feedback_controller.convergence_tracker.get_converged_nodes())
                                    local_converged_in_train = sum((1 for nid in converged_set if nid in train_nodes_cpu_set))
                                    local_remaining = len(train_nodes_cpu_set) - local_converged_in_train

                                    # 全局求和（各 rank 训练集互不重叠时，该和即全局训练集中已收敛数）
                                    global_converged = torch.tensor([local_converged_in_train], device=env.device, dtype=torch.long)
                                    if dist.is_available() and dist.is_initialized():
                                        dist.all_reduce(global_converged, op=dist.ReduceOp.SUM)

                                    env.logger.log(
                                        f"Epoch {epoch} | Local converged in-train: {local_converged_in_train}, Remaining: {local_remaining}",
                                        rank=0
                                    )
                                    if dist.is_available() and dist.is_initialized():
                                        env.logger.log(
                                            f"Epoch {epoch} | Global converged in-train: {int(global_converged.item())}",
                                            rank=0
                                        )
                                except Exception:
                                    pass

                            with env.timer.timing_cuda('opt.step'):
                                optimizer.step()
                            env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)
                    else:
                        # 非反馈模式，使用所有训练节点
                        batch_nodes = train_nodes
                        with env.timer.timing('loss'):
                            if g.labels.dim() == 1:
                                batch_losses = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes])
                            else:
                                batch_losses = loss_func(outputs[batch_nodes], g.local_labels[batch_nodes]).mean(dim=1)
                            loss = batch_losses.mean()

                        with env.timer.timing_cuda('backward'):
                            loss.backward()
                        
                        with env.timer.timing_cuda('opt.step'):
                            optimizer.step()
                        env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)
                else:
                    env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
                    with env.timer.timing('loss'):
                        loss = (outputs * 0).sum()
                    with env.timer.timing_cuda('backward'):
                        loss.backward()


        if epoch%10==0 or epoch==args.epoch-1:
            with env.timer.timing('eval.gather'):
                all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                mask = g.train_mask
                env.logger.log(f'Epoch: {epoch:03d}', f1(g.labels[mask], torch.sigmoid(all_outputs[mask])), rank=0)
            else:
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)

        # 性能优化：减少状态打印频率
        if feedback_system.is_enabled() and epoch % max(10, base_feedback_every * 2) == 0:
            status = feedback_system.get_status()
            env.logger.log(f"Epoch {epoch} | Feedback System Status: {status}", rank=0)
            
            # 动态调整采样策略
            if 'convergence_rate' in status:
                feedback_system.adjust_strategy(epoch, status['convergence_rate'])
                
        # 注释：基于收敛率的早停机制已移除，现在只使用基于本地节点全收敛的早停机制


def train(g, env, args):
    """原始训练函数（保持兼容性，加入时间统计）"""
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if g.labels.dim()==1:
        loss_func = nn.CrossEntropyLoss()
    elif g.labels.dim()==2:
        loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    
    for epoch in range(args.epoch):
        with env.timer.timing('epoch'):
            with autocast(env.half_enabled):
                with env.timer.timing('forward'):
                    outputs = model(g.features)
                optimizer.zero_grad()
                if g.local_labels[g.local_train_mask].size(0) > 0:
                    with env.timer.timing('loss'):
                        loss = loss_func(outputs[g.local_train_mask], g.local_labels[g.local_train_mask])
                else:
                    env.logger.log('Warning: no training nodes in this partition! Backward fake loss.')
                    with env.timer.timing('loss'):
                        loss = (outputs * 0).sum()
            with env.timer.timing_cuda('backward'):
                loss.backward()
            with env.timer.timing_cuda('opt.step'):
                optimizer.step()
            env.logger.log("Epoch {:05d} | Loss {:.4f}".format(epoch, loss.item()), rank=0)

        if epoch%10==0 or epoch==args.epoch-1:
            with env.timer.timing('eval.gather'):
                all_outputs = env.all_gather_then_cat(outputs)
            if g.labels.dim()>1:
                mask = g.train_mask
                env.logger.log(f'Epoch: {epoch:03d}', f1(g.labels[mask], torch.sigmoid(all_outputs[mask])), rank=0)
            else:
                acc = lambda mask: all_outputs[mask].max(1)[1].eq(g.labels[mask]).sum().item()/mask.sum().item()
                env.logger.log(f'Epoch: {epoch:03d}, Train: {acc(g.train_mask):.4f}, Val: {acc(g.val_mask):.4f}, Test: {acc(g.test_mask):.4f}', rank=0)


def main(env, args):
    env.csr_enabled = False
    env.csr_enabled = True

    env.half_enabled = True
    env.half_enabled = False

    env.logger.log('proc begin:', env)

    with env.timer.timing('total'):
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

        # 初始化信息互馈系统（增加候选池与统计开关）
        enable_feedback = getattr(args, 'enable_feedback', True)
        sampler_candidate_pool_size = getattr(args, 'candidate_pool_size', None)
        keep_sampling_stats = getattr(args, 'keep_sampling_stats', False)
        similarity_threshold = getattr(args, 'similarity_threshold', None)
        patience = getattr(args, 'patience', None)
        min_epochs = getattr(args, 'min_epochs', None)
        sampling_strategy = getattr(args, 'sampling_strategy', None)
        
        # 新增：性能优化参数
        feedback_batch_cap = getattr(args, 'feedback_batch_cap', None)
        use_simple_convergence = getattr(args, 'use_simple_convergence', True)
        
        # 早停机制参数
        enable_early_stopping = getattr(args, 'enable_early_stopping', False)
        early_stop_threshold = getattr(args, 'early_stop_threshold', 0.02)
        early_stop_patience = getattr(args, 'early_stop_patience', 3)
        min_epochs_before_stop = getattr(args, 'min_epochs_before_stop', 5)
        
        feedback_system = InfoFeedbackSystem(
            g.num_nodes, env.device, enable_feedback,
            sampler_candidate_pool_size=sampler_candidate_pool_size,
            keep_sampling_stats=keep_sampling_stats,
            similarity_threshold=similarity_threshold,
            patience=patience,
            min_epochs=min_epochs,
            sampling_strategy=sampling_strategy,
            feedback_batch_cap=feedback_batch_cap,
            use_simple_convergence=use_simple_convergence,
            enable_early_stopping=enable_early_stopping,
            early_stop_threshold=early_stop_threshold,
            early_stop_patience=early_stop_patience,
            min_epochs_before_stop=min_epochs_before_stop
        )

        if enable_feedback:
            env.logger.log('Starting training with Info-Feedback system', rank=0)
            train_with_feedback(g, env, args, feedback_system)
        else:
            env.logger.log('Starting standard training without feedback', rank=0)
            from dist_train import train as train_plain
            train_plain(g, env, args)

    if env.rank == 0:
        print(f"Model: {args.model} layers: {args.nlayers} nprocs {args.nprocs}")
        if enable_feedback:
            print("Info-Feedback system was enabled")
        else:
            print("Info-Feedback system was disabled")

    env.logger.log(env.timer.summary_all(), rank=0)
    if enable_feedback:
        final_status = feedback_system.get_status()
        env.logger.log(f"Final Feedback System Status: {final_status}", rank=0)
