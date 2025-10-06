# Copyright 2021, Zhao CHEN
# All rights reserved.

import os
import torch
# from .friendster import FriendSterDataset

# 设置数据的根目录、DGL 数据集目录和 PyTorch Geometric 数据集目录
data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
dgl_root = os.path.join(data_root, 'dgl_datasets')
pyg_root = os.path.join(data_root, 'pyg_datasets')
user_root = os.path.join("/home/lzl/nfs.d/dataset/graph_embedding/LinkPrediction/train_data/")
rmat_root = os.path.join("/home/lzl/nfs.d/dataset/graph_embedding/graph_data/")
# 确保目录存在，如果不存在则创建
for path in [data_root, dgl_root, pyg_root]:
    os.makedirs(path, exist_ok=True)

# --- 新增：用户自定义数据集配置字典 ---
USER_DATASET_CONFIG = {
    'com': {
        'edgelist_path': os.path.join(user_root, 'com_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'LJ': {
        'edgelist_path': os.path.join(user_root, 'LJ_srt_wei_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 16,
    },
    'soc': {
        'edgelist_path': os.path.join(user_root, 'soc_srt_wei_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'wv': {
        'edgelist_path': os.path.join(user_root, 'wv_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'ytb': {
        'edgelist_path': os.path.join(user_root, 'ytb_srt_weg_cn_train.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'twt': {
        'edgelist_path': os.path.join(user_root, 'twt.edge'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'rmat5': {
        'edgelist_path': os.path.join(rmat_root, 'rmat5_srt.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'rmat6': {
        'edgelist_path': os.path.join(rmat_root, 'rmat6_srt.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'rmat7': {
        'edgelist_path': os.path.join(rmat_root, 'rmat7_srt.txt'),
        'feature_dim': 128, 'hidden_dim': 128, 'num_classes': 10,
    },
    'rmat8': {
        'edgelist_path': os.path.join(rmat_root, 'rmat8_srt.txt'),
        'feature_dim': 10, 'hidden_dim': 5, 'num_classes': 2,
    },
    'rmat9': {
        'edgelist_path': os.path.join(rmat_root, 'rmat9_srt.txt'),
        'feature_dim': 5, 'hidden_dim': 3, 'num_classes': 2,
    },
}

# 保存图数据集的函数
def save_dataset(edge_index, features, labels, train_mask, val_mask, test_mask, num_nodes, num_edges, num_classes, name):
    if name.startswith('a_quarter'):
        # 如果数据集名字以 'a_quarter' 开头，保留图的四分之一节点数的数据
        max_node = num_nodes//4
        smaller_mask = (edge_index[0]<max_node) & (edge_index[1]<max_node)

        edge_index = edge_index[:, smaller_mask].clone()
        features = features[:max_node].clone()
        labels = labels[:max_node].clone()
        train_mask = train_mask[:max_node].clone()
        val_mask = val_mask[:max_node].clone()
        test_mask = test_mask[:max_node].clone()
        num_nodes = max_node
        num_edges = edge_index.size(1)
    # 构建保存路径
    path = os.path.join(data_root, name+'.torch')
    # 保存数据集至指定路径
    torch.save({"edge_index": edge_index.int(), "features": features, "labels": labels.char(),
                "train_mask": train_mask.bool(), 'val_mask': val_mask.bool(), 'test_mask': test_mask.bool(),
                "num_nodes": num_nodes, 'num_edges': num_edges, 'num_classes': num_classes}, path)

# 加载图数据集的函数
def load_dataset(name):
    # 构建加载路径
    path = os.path.join(data_root, name+'.torch')
    if not os.path.exists(path):
        # 如果文件不存在，调用 prepare_dataset(name) 准备数据集
        prepare_dataset(name)
    # 加载数据集
    return torch.load(path)

# DGL 数据集准备函数
def prepare_dgl_dataset(dgl_name, tag):
    import dgl
    # 定义数据集源
    dataset_sources = {'cora': dgl.data.CoraGraphDataset, 'reddit': dgl.data.RedditDataset}
    # 加载DGL数据集
    dgl_dataset: dgl.data.DGLDataset = dataset_sources[dgl_name](raw_dir=dgl_root)
    # 获取图对象
    g = dgl_dataset[0]
    # 将图的邻接矩阵转换为PyTorch张量
    edge_index = torch.stack(g.adj_sparse('coo'))
    print('dgl dataset', dgl_name, 'loaded')
    # 保存数据集至指定路径
    save_dataset(edge_index, g.ndata['feat'], g.ndata['label'],
                 g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'],
                 g.num_nodes(), g.num_edges(), dgl_dataset.num_classes, tag)

def prepare_user_dataset_from_edgelist(tag):
    """
    从边列表文件生成完整的数据集，包含随机特征和标签，并将所有节点设置为训练集，
    最后保存为 NeutronTP 所需的 .torch 格式。
    """
    import numpy as np
    config = USER_DATASET_CONFIG[tag]
    edgelist_path = config['edgelist_path']
    num_classes = config['num_classes']
    feature_dim = config['feature_dim']
    # 加载边列表
    print(f"Loading edge list from: {edgelist_path}")
    if not os.path.exists(edgelist_path):
        raise FileNotFoundError(f"Edgelist file not found at: {edgelist_path}")
        
    edge_data = np.loadtxt(edgelist_path, dtype=int)
    
    # 检查并处理带权重的边列表
    if edge_data.shape[1] == 3:
        print("Detected weighted edge list (3 columns). Using only source and target nodes.")
        edges_list = edge_data[:, :2]
    elif edge_data.shape[1] == 2:
        edges_list = edge_data
    else:
        raise ValueError("Edge list file should have 2 or 3 columns.")

    # 获取节点和边的数量
    num_nodes = int(edges_list.max()) + 1
    num_edges = edges_list.shape[0]
    print(f"Detected {num_nodes} nodes and {num_edges} edges.")

    # 生成随机特征
    print(f"Generating {feature_dim}-dimensional features...")
    features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    # 特征归一化
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norm == 0, 1, norm)

    # 生成随机标签
    print(f"Generating {num_classes} classes of random labels...")
    labels = np.random.randint(0, num_classes, size=num_nodes)

    print("Splitting nodes into training (50%), validation (10%), and test (40%) sets...")
    
    # 创建一个随机排列的索引
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)

    # 定义划分比例
    train_ratio = 0.5
    val_ratio = 0.1
    
    # 计算各个集合的大小
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    # 分配索引
    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]
    
    # 创建掩码
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    print(f"Train nodes: {train_mask.sum()}, Validation nodes: {val_mask.sum()}, Test nodes: {test_mask.sum()}")
    
    # 将Numpy数组转换为PyTorch张量以符合 save_dataset 的要求
    edge_index_tensor = torch.from_numpy(edges_list.T).int()
    features_tensor = torch.from_numpy(features)
    labels_tensor = torch.from_numpy(labels)
    train_mask_tensor = torch.from_numpy(train_mask)
    val_mask_tensor = torch.from_numpy(val_mask)
    test_mask_tensor = torch.from_numpy(test_mask)

    print("Saving dataset in NeutronTP format...")
    # 调用系统中已有的 save_dataset 函数
    save_dataset(
        edge_index=edge_index_tensor,
        features=features_tensor,
        labels=labels_tensor,
        train_mask=train_mask_tensor,
        val_mask=val_mask_tensor,
        test_mask=test_mask_tensor,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_classes=num_classes,
        name=tag
    )
    print(f"Dataset '{tag}' saved successfully.")
    


# DGL 数据集准备函数
def prepare_tsp_dataset(dgl_name, tag):
    # 定义数据集源
    dataset = FriendSterDataset()
    g = dataset[0]
    # 将图的邻接矩阵转换为PyTorch张量
    edge_index = torch.stack(g.adj_sparse('coo'))
    print('dgl dataset', dgl_name, 'loaded')
    # 保存数据集至指定路径
    save_dataset(edge_index, g.ndata['feat'], g.ndata['label'],
                 g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask'],
                 g.num_nodes(), g.num_edges(), dataset.num_classes, tag)
    

# PyTorch Geometric 数据集准备函数
def prepare_pyg_dataset(pyg_name, tag):
    import torch_geometric
    import ogb.nodeproppred
    # 定义数据集源
    dataset_sources = {'reddit': torch_geometric.datasets.Reddit,
                       'flickr': torch_geometric.datasets.Flickr,
                       'yelp': torch_geometric.datasets.Yelp,
                        'amazon-products': torch_geometric.datasets.AmazonProducts,
                        }
    print('pyg dataset root:', pyg_root)
    # 加载 PyTorch Geometric 数据集
    pyg_dataset: torch_geometric.data.Dataset = dataset_sources[pyg_name](root=os.path.join(pyg_root, pyg_name))
    print('pyg dataset', pyg_name, 'loaded')
    # 获取数据集中的第一个数据对象
    data: torch_geometric.data.Data = pyg_dataset[0]
    # 保存数据集至指定路径
    save_dataset(data.edge_index, data.x, data.y,
                 data.train_mask, data.val_mask, data.test_mask,
                 data.num_nodes, data.num_edges, pyg_dataset.num_classes, tag)


def prepare_ogb_dataset(pyg_name, tag):
    import torch_geometric
    import ogb.nodeproppred
    # 定义数据集源
    dataset_source =  ogb.nodeproppred.PygNodePropPredDataset
    #  dataset_sources = { 'ogbn-products', 'ogbn-arxiv', 'ogbn-papers100M', }
    # 加载 OGB 数据集
    dataset = dataset_source(root=os.path.join(pyg_root, pyg_name), name=pyg_name)
    print('ogb dataset', pyg_name, 'loaded')
    # 获取数据集中的第一个数据对象
    data: torch_geometric.data.Data = dataset[0]
    # 如果标签是 2D 的，将其转换为 1D
    if data.y.dim() == 2 and data.y.size(1) == 1:
        label_1d = data.y.view(-1)
    else:
        label_1d = data.y

    if pyg_name == 'ogbn-papers100M':
        # 自定义 5:1:4 划分
        generator = torch.Generator()
        generator.manual_seed(42)
        perm = torch.randperm(data.num_nodes, generator=generator)
        train_size = int(data.num_nodes * 0.5)
        val_size = int(data.num_nodes * 0.1)
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]
        test_idx = perm[train_size + val_size:]
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        del perm
    else:
        # 默认使用 OGB 提供的划分
        split_idx = dataset.get_idx_split()
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros_like(train_mask)
        test_mask = torch.zeros_like(train_mask)
        train_mask[split_idx['train']] = True
        val_mask[split_idx['valid']] = True
        test_mask[split_idx['test']] = True

    # 保存数据集至指定路径
    save_dataset(data.edge_index, data.x, label_1d,
                 train_mask, val_mask, test_mask,
                 data.num_nodes, data.num_edges, dataset.num_classes, tag)


# 选择并准备指定数据集的函数
def prepare_dataset(tag):
        # 检查是否为用户自定义的数据集
    if tag in USER_DATASET_CONFIG:
        return prepare_user_dataset_from_edgelist(tag)

    if tag=='reddit':
        return prepare_pyg_dataset('reddit', tag)
    elif tag=='flickr':
        return prepare_pyg_dataset('flickr', tag)
    elif tag == 'yelp':  # graphsaints
        return prepare_pyg_dataset('yelp', tag)
    elif tag=='amazon-products':  # graphsaints
        return prepare_pyg_dataset('amazon-products', tag)
    elif tag=='cora':
        return prepare_dgl_dataset('cora', tag)
    elif tag=='reddit_reorder':
        return prepare_dgl_dataset('reddit', tag)
    elif tag=='a_quarter_reddit':
        return prepare_pyg_dataset('reddit', tag)
    elif tag=='ogbn-products':
        return prepare_ogb_dataset('ogbn-products', tag)
    elif tag == 'ogbn-arxiv':
        return prepare_ogb_dataset('ogbn-arxiv', tag)
    elif tag == 'ogbn-100m':
        return prepare_ogb_dataset('ogbn-papers100M', tag)
    elif tag == 'friendster':
        return prepare_tsp_dataset('friendster', tag)
    else:
        print('no such dataset', tag)


def check_edges(edge_index, num_nodes):
    print(f'edges {edge_index[0].size(0)} nodes:{num_nodes}')
    num_parts = 4
    split_size = num_nodes//num_parts
    first_limit = split_size
    last_limit = num_nodes - split_size

    fist_size = (edge_index[0] < first_limit).sum()
    last_size = (edge_index[0] > last_limit).sum()
    print(f'first block {fist_size} last block {last_size}')

    mask_first = (edge_index[0] < first_limit) & (edge_index[1] < first_limit)
    mask_last = (edge_index[0] > last_limit) & (edge_index[1] > last_limit)

    fist_size = edge_index[0][mask_first].size(0)
    last_size = edge_index[0][mask_last].size(0)
    print(f'first p {fist_size} last p {last_size}')


def main():
    prepare_dataset('ogbn-100m')
    return
    prepare_dataset('ogbn-arxiv')
    prepare_dataset('ogbn-products')
    return
    for dataset_name in ['cora', 'reddit', 'flickr', 'yelp', 'flickr', 'a_quarter_reddit','amazon-products']:
        prepare_dataset(dataset_name)
    return


if __name__ == '__main__':
    main()
