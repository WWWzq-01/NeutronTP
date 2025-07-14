import os
import argparse
import torch
import socket
import dist_utils
import dist_train
import torch.distributed as dist

# 根据IP地址自动判断rank值的函数
def get_rank_from_ip(master_addr):
    # 获取当前机器的所有IP地址
    hostname = socket.gethostname()
    local_ips = socket.gethostbyname_ex(hostname)[2]
    
    # 如果当前机器是master节点，rank为0
    if master_addr in local_ips:
        return 0
    
    # 如果配置了NODE_RANK环境变量，则使用它
    if 'NODE_RANK' in os.environ:
        return int(os.environ['NODE_RANK'])
    
    # 如果配置了明确的rank映射，则使用它
    ip_rank_map = {
        '192.168.6.130': 0,  # master节点
        '192.168.6.129': 1,
        '192.168.6.128': 2,
        '192.168.6.127': 3, 
        '192.168.6.126': 4,
        '192.168.6.125': 5,
        '192.168.6.124': 6,  
        '192.168.6.123': 7,
        # 可以添加更多节点的IP和对应rank
    }
    
    for ip in local_ips:
        if ip in ip_rank_map:
            return ip_rank_map[ip]
    
    # 如果无法确定rank，给出警告并默认为0
    print(f"警告: 无法根据IP自动确定rank值。当前机器IP: {local_ips}")
    print("请使用环境变量NODE_RANK指定rank值或修改代码中的ip_rank_map。")
    return 0

# 定义一个函数，用于包装分布式训练的设置和启动
def process_wrapper(rank, args, func):
    # for single machine
    # os.environ['MASTER_ADDR'] = '127.0.0.1'  #设置分布式训练的主节点，127.0.0.1默认是本地节点
    # os.environ['MASTER_PORT'] = '29501'    #端口号
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo' #NCCL网络接口名称，'lo'通常表示本地回环接口，GPU通信将通过本地主机进行
    # env = dist_utils.DistEnv(rank, args.nprocs, args.backend)
    # env.half_enabled = True
    # env.csr_enabled = True
    # for multi machine
    master_addr = '192.168.6.130'
    os.environ['MASTER_ADDR'] = master_addr  #设置分布式训练的主节点，127.0.0.1默认是本地节点
    os.environ['MASTER_PORT'] = '29500'    #端口号
    os.environ['NCCL_SOCKET_IFNAME'] = 'ens17f0' #通信接口
    # 创建 DistEnv 对象，该对象封装了分布式训练的环境信息
    # 获取节点rank值
    node_rank = get_rank_from_ip(master_addr)
    # 计算全局rank值 (node_rank * 每节点GPU数 + 当前进程的本地rank)
    global_rank = node_rank * args.nprocs + rank
    
    print(f"当前进程信息: 节点rank={node_rank}, 本地rank={rank}, 全局rank={global_rank}")
    # rank = 0 #多机下，要手动指定rank值
    env = dist_utils.DistEnv(global_rank, args.nnodes, args.backend)
    env.half_enabled = True
    env.csr_enabled = True


    # 调用传入的 func 函数，开始分布式训练
    func(env, args)


if __name__ == "__main__":
    num_GPUs = torch.cuda.device_count()
    parser = argparse.ArgumentParser() 
    # parser.add_argument("--nprocs", type=int, default=num_GPUs if num_GPUs>1 else 8)
    #single GPU
    parser.add_argument("--nprocs", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=20)
    # parser.add_argument("--backend", type=str, default='gloo')
    parser.add_argument("--backend", type=str, default='nccl' if num_GPUs>1 else 'gloo') 
    # parser.add_argument("--dataset", type=str, default='ogbn-100m')
    # parser.add_argument("--dataset", type=str, default='friendster')
    # parser.add_argument("--dataset", type=str, default='reddit')
    parser.add_argument("--dataset", type=str, default='cora')
    # parser.add_argument("--model", type=str, default='DecoupleGCN')
    # parser.add_argument("--model", type=str, default='GCN')
    # parser.add_argument("--model", type=str, default='TensplitGCN')
    # parser.add_argument("--model", type=str, default='TensplitGCNLARGE')
    # parser.add_argument("--model", type=str, default='TensplitGCNSWAP')
    # parser.add_argument("--model", type=str, default='TensplitGCNCPU')
    # parser.add_argument("--model", type=str, default='TensplitGATLARGE')
    # parser.add_argument("--model", type=str, default='GAT')
    parser.add_argument("--model", type=str, default='TensplitGAT')
    args = parser.parse_args()
    process_args = (args, dist_train.main)
    # 启动多个进程进行分布式训练
    torch.multiprocessing.spawn(process_wrapper, process_args, args.nprocs)
