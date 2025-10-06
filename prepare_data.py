# Copyright 2021, Zhao CHEN
# All rights reserved.
import coo_graph
import argparse
# import time

def main():
    cached = True
    dataset = 'ogbn-100m'
    npart = 8
    # r = coo_graph.COO_Graph_Full('arxiv')
    # r = coo_graph.COO_Graph('reddit')
    # r = coo_graph.COO_Graph('test', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('flickr', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('reddit', full_graph_cache_enabled=cached)
    # r = coo_graph.COO_Graph('ogbn-arxiv', full_graph_cache_enabled=cached)
    
    r = coo_graph.COO_Graph(dataset, full_graph_cache_enabled=cached)
    
    # start_time = time.perf_counter()
    # r.partition(npart)
    
    # end_time = time.perf_counter()
            
    # duration = end_time - start_time
    # print(f"r.partition({npart:d}) took {duration:.4f} seconds to run.") # 打印运行时间，保留4位小数
                            
    # return
    # r.partition(1)
    # r.partition(2)
    # r.partition(4)
    r.partition(8)
    return
    # for name in ['amazon-products', 'ogbn-products']:
    # for name in ['ogbn-arxiv', 'ogbn-products']:
    #     r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
    #     r.partition(4)
    #     r.partition(8)
    #     print(r)
    # return
    # for name in ['reddit', 'yelp', 'flickr', 'cora', 'ogbn-arxiv']:
    #     r = coo_graph.COO_Graph(name, full_graph_cache_enabled=cached)
    #     r.partition(8)
    #     r.partition(4)
    #     print(r)
    # return


if __name__ == '__main__':
    main()
