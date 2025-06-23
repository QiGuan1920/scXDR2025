# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:19:29 2024

@author: 78760
"""

import dgl
import torch
import numpy as np
from dgl.data.utils import load_graphs

def Edge_rename(graph, x):
    edges_data = {}
    for canonical_etype in graph.canonical_etypes:
        srctype, etype, dsttype = canonical_etype
        src, dst = graph.edges(etype=canonical_etype)
        
        if etype == 'CD':
            new_etype = ('cell', 'CD' + x, 'drug')
        elif etype == 'DC':
            new_etype = ('drug', 'DC' + x, 'cell')
        else:
            new_etype = canonical_etype
            
        edges_data[new_etype] = (src, dst)
    
    renamed_graph = dgl.heterograph({etype: (edges_data[etype][0], edges_data[etype][1]) for etype in edges_data})
    return renamed_graph

def load_and_prepare_graphs(S1, S2, base_path):
    source_basic_graph, _ = load_graphs(f'{base_path}/{S1}_basic.bin')
    source_pos_graph, _ = load_graphs(f'{base_path}/{S1}_pos.bin')
    source_neg_graph, _ = load_graphs(f'{base_path}/{S1}_neg.bin')

    target_basic_graph, _ = load_graphs(f'{base_path}/{S2}_basic.bin')
    target_pos_graph, _ = load_graphs(f'{base_path}/{S2}_pos.bin')
    target_neg_graph, _ = load_graphs(f'{base_path}/{S2}_neg.bin')

    source_basic_graph = source_basic_graph[0]
    source_pos_graph = source_pos_graph[0]
    source_neg_graph = source_neg_graph[0]
    
    target_basic_graph = target_basic_graph[0]
    target_pos_graph = target_pos_graph[0]
    target_neg_graph = target_neg_graph[0]
    
    source_pos_graph = Edge_rename(source_pos_graph, 'pos')
    source_neg_graph = Edge_rename(source_neg_graph, 'neg')
    target_pos_graph = Edge_rename(target_pos_graph, 'pos')
    target_neg_graph = Edge_rename(target_neg_graph, 'neg')
    
    source_features = {node_type: source_basic_graph.nodes[node_type].data['features'] for node_type in source_basic_graph.ntypes}
    target_features = {node_type: target_basic_graph.nodes[node_type].data['features'] for node_type in target_basic_graph.ntypes}

    return source_basic_graph, source_pos_graph, source_neg_graph, target_basic_graph, target_pos_graph, target_neg_graph, source_features, target_features


def safe_min_max_normalize(features):
    if features.size(0) == 1:
        return features
    min_val = torch.min(features, dim=0)[0]
    max_val = torch.max(features, dim=0)[0]
    range = max_val - min_val
    range[range == 0] = 1
    normalized_features = (features - min_val) / range
    return normalized_features










