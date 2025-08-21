import os
import argparse
import torch

import dist_utils
import dist_train_with_feedback
import torch.distributed as dist


# 单机环境包装与启动
def process_wrapper(rank, args, func):
    # single machine
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'

    env = dist_utils.DistEnv(rank, args.nprocs, args.backend)
    env.half_enabled = True
    env.csr_enabled = True

    # 启动训练
    func(env, args)


if __name__ == "__main__":
    num_GPUs = torch.cuda.device_count()
    parser = argparse.ArgumentParser()

    # 训练与模型通用参数
    parser.add_argument("--nprocs", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--backend", type=str, default='nccl' if num_GPUs>1 else 'gloo')
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--model", type=str, default='TensplitGAT')

    # 信息互馈相关参数（单机）
    parser.add_argument("--enable_feedback", action="store_true", help="启用信息互馈系统")
    parser.add_argument("--disable_feedback", action="store_true", help="禁用信息互馈系统，使用标准训练")
    parser.add_argument("--batch_size", type=int, default=1000, help="训练batch大小（启用反馈时生效）")

    # 相似性收敛参数
    parser.add_argument("--similarity_threshold", type=float, default=0.9, help="嵌入相似性收敛阈值")
    parser.add_argument("--patience", type=int, default=2, help="相似性判断耐心窗口")
    parser.add_argument("--min_epochs", type=int, default=1, help="开始判断的最小epoch")

    # 采样策略
    parser.add_argument("--sampling_strategy", type=str, default='convergence_aware',
                        choices=['no_importance', 'random', 'convergence_aware', 'adaptive_importance'],
                        help="采样策略。no_importance=均匀采样且过滤已收敛节点")

    # 性能优化参数
    parser.add_argument("--feedback_every", type=int, default=5, help="每多少个epoch进行一次反馈")
    parser.add_argument("--sampler_candidate_pool_size", type=int, default=100000, help="采样候选池大小；设为0表示禁用")
    parser.add_argument("--keep_sampling_stats", action="store_true", help="保留采样统计（可能增加内存开销）")

    # 新增：性能优化参数
    parser.add_argument('--feedback_batch_cap', type=int, default=20000,
                       help='反馈批大小上限（用于限制反馈处理的节点数量）')
    parser.add_argument('--ema_decay', type=float, default=0.8,
                       help='EMA衰减率（0.8-0.95，越小更新越快）')
    parser.add_argument('--use_simple_convergence', action='store_true', default=True,
                       help='使用简单收敛模式（更快，推荐）')

    args = parser.parse_args()

    # 规范化 feedback 开关
    if args.disable_feedback:
        args.enable_feedback = False
    elif not args.disable_feedback:
        args.enable_feedback = True

    # 规范化候选池
    if args.sampler_candidate_pool_size is not None and args.sampler_candidate_pool_size <= 0:
        args.sampler_candidate_pool_size = None

    print(args)

    process_args = (args, dist_train_with_feedback.main)
    torch.multiprocessing.spawn(process_wrapper, process_args, args.nprocs)
