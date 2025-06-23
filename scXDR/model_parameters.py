# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:59:06 2024

@author: 78760
"""

from models import *
from data_processing import *



def prepare_data_fold(source_pos_graph, source_neg_graph, num_folds=5):

    # 获取源图中的正边数量
    num_pos_edges = source_pos_graph.number_of_edges(etype=('drug', 'DCpos', 'cell'))
    # 获取源图中的负边数量
    num_neg_edges = source_neg_graph.number_of_edges(etype=('drug', 'DCneg', 'cell'))
    # 计算总和
    num_edges = num_pos_edges + num_neg_edges
    # 生成边的索引
    idx = np.arange(num_edges)
    # 随机打乱索引
    np.random.shuffle(idx)
    # 将边等分成五份
    folds = np.array_split(idx, num_folds)

    return folds



def prepare_seed(seed_number = 10):
    
    seed = seed_number    ### 种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def prepare_model():
    
    max_edges_per_type = 2000  # 根据需要设置最大边数目
    
    # 初始化模型
    in_feats = {'drug': 167, 'cell': 5000, 'target': 147}
    # hidden_feats = 64
    # out_feats = 64
    
    rel_names2 = [('drug', 'DT', 'target'),('target', 'TD', 'drug'),('cell', 'CT', 'target'), ('target', 'TC', 'cell'), ('cell', 'CC1', 'cell'),('cell', 'CC2', 'cell'), ('target', 'TT1', 'target'), ('target', 'TT2', 'target')]
    
    
    gcn_model = HeteroRGCN(in_feats, 64, 64, rel_names2)
    
    feature_aligners = FeatureAlignAutoEncoder(64, 32)
    
    edge_aligners = EdgeAlignAutoEncoder(64 * 2, 32)
    
    structure_aligners = StructureAlignAutoEncoder(32 * 3, 32)
    
    link_predictor = HeteroDotProductPredictor()
    
    predictor = HeteroMLPPredictor(32,1)
    
    
    optimizer = torch.optim.Adam(
        list(gcn_model.parameters()) + 
        list(feature_aligners.parameters()) +
        list(edge_aligners.parameters()) +
        list(structure_aligners.parameters()) +
        list(predictor.parameters()), lr=0.001)
    
    
    criterion = nn.MSELoss()
    criterion_edge = torch.nn.BCEWithLogitsLoss()
    
    return gcn_model, feature_aligners, edge_aligners, structure_aligners, link_predictor, predictor, optimizer, criterion, criterion_edge, max_edges_per_type






