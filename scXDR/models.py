import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
import pandas as pd
from dgl.nn.pytorch import RelGraphConv
import dgl.nn as dglnn
from dgl import function as fn
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
from dgl.data.utils import save_graphs, load_graphs
import random


class HeteroRGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats[rel[0]], hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroMLPPredictor(nn.Module):
    def __init__(self, in_dims, n_classes):
        super().__init__()
        self.W1 = nn.Linear(in_dims * 2, in_dims)
        self.W2 = nn.Linear(in_dims, n_classes)

    def apply_edges(self, edges):
        x = torch.cat([edges.src['h'], edges.dst['h']], 1)
        y = torch.relu(self.W1(x))
        y = self.W2(y)
        
        return {'score': y}

    def forward(self, graph, h, etype):
        # h contains the node representations
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edata['score']


class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    
    # Get edges that do not exist in the graph
    neg_src = []
    neg_dst = []
    edges_set = set(zip(src.tolist(), dst.tolist()))
    
    for i in range(len(src)):
        while True:
            # Generate random negative samples
            neg_dst_node = torch.randint(0, graph.number_of_nodes(vtype), (k,))
            for j in range(k):
                if (src[i], neg_dst_node[j].item()) not in edges_set:
                    neg_src.append(src[i].item())
                    neg_dst.append(neg_dst_node[j].item())
                    break
            if len(neg_src) == (i + 1) * k:
                break
    
    # Create a negative graph
    return dgl.heterograph(
        {etype: (torch.tensor(neg_src), torch.tensor(neg_dst))},
        num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})


def linkrecon_loss(pos_score, neg_score):
    # Determine the smaller size
    min_size = min(pos_score.size(0), neg_score.size(0))
    
    # Randomly sample to make the sizes equal
    pos_indices = torch.randperm(pos_score.size(0))[:min_size]
    neg_indices = torch.randperm(neg_score.size(0))[:min_size]
    
    # Select samples with equal size based on indices
    pos_score = pos_score[pos_indices]
    neg_score = neg_score[neg_indices]
    # Calculate margin loss
    margin_loss = 1 + neg_score.unsqueeze(1) - pos_score.unsqueeze(1)
    
    # Use clamp to restrict values smaller than 0 to 0
    margin_loss = torch.clamp(margin_loss, min=0)
    
    # Calculate the mean of the loss
    loss = torch.mean(margin_loss)
    
    return loss


def lossloss(linkmodel, graph, k, embeddings, etype):
    
    negative_graph = construct_negative_graph(graph, k, etype)
    pos_score = linkmodel(graph,  embeddings, etype)
    neg_score = linkmodel(negative_graph, embeddings, etype)
    link_lossloss = linkrecon_loss(pos_score, neg_score)
    
    return link_lossloss


def Fold(folds, fold):
    
    test_idx = folds[fold]
    train_idx = np.concatenate([folds[i] for i in range(5) if i != fold])
    
    return test_idx, train_idx


# Define a node feature alignment autoencoder
class FeatureAlignAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureAlignAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define an edge feature alignment autoencoder
class EdgeAlignAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(EdgeAlignAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define a structural alignment autoencoder
class StructureAlignAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(StructureAlignAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Get high-density points
def get_high_density_points(features, num_points):
    if features.size(0) == 0:
        return features  # If there are no features, return empty features

    kde = KernelDensity(kernel='gaussian', bandwidth=0.5)
    kde.fit(features.detach().numpy())
    densities = kde.score_samples(features.detach().numpy())
    
    # Handle the case where the requested number of high-density points exceeds the number of features
    num_points = min(num_points, features.size(0))
    
    indices = torch.tensor(densities).argsort(descending=True)[:num_points]
    return features[indices]


# Calculate the alignment loss
def calculate_alignment_loss(source_feats, target_feats):
    euclidean_loss = F.mse_loss(source_feats, target_feats)
    cos_sim = F.cosine_similarity(source_feats, target_feats, dim=-1)
    cosine_loss = 1 - torch.mean(cos_sim)
    return euclidean_loss + cosine_loss


def get_common_path_types(source_meta_paths, target_meta_paths):
    """
    Compute the intersection of the source domain and target domain meta-path types.
    
    Parameters:
    - source_meta_paths: A list of meta-path samples from the source domain
    - target_meta_paths: A list of meta-path samples from the target domain
    
    Returns:
    - common_path_types: A set of common meta-path types between the source and target domains
    """
    # Extract the meta-path types from the source and target domains
    source_path_types = set(sample[0] for sample in source_meta_paths)
    target_path_types = set(sample[0] for sample in target_meta_paths)

    # Take the intersection to ensure loss is computed only for common meta-path types
    common_path_types = source_path_types.intersection(target_path_types)
    return common_path_types


def sample_meta_paths(graph, node_representations, two_hop_paths, common_path_types=None):
    """
    Generate meta-path samples and their features from the graph, with optional filtering by common path types.
    
    Parameters:
    - graph: The graph object
    - node_representations: A dictionary of node features
    - two_hop_paths: A list of meta-path types
    - common_path_types: The intersection of meta-path types (optional), only samples from the intersection will be generated
    
    Returns:
    - meta_path_samples: A list of samples containing the meta-path type and its features
    """
    meta_path_samples = []
    for start_node_type, relation1, mid_node_type, relation2, end_node_type in two_hop_paths:
        # Define the meta-path type
        path_type = (start_node_type, relation1, mid_node_type, relation2, end_node_type)
        # Check if the meta-path type is in the intersection
        if common_path_types and path_type not in common_path_types:
            continue  # Skip meta-path types not in the intersection

        # Generate meta-path samples and their features
        start_nodes = graph.nodes(start_node_type)
        metapath = [relation1, relation2]
        traces, _ = dgl.sampling.random_walk(graph, nodes=start_nodes, metapath=metapath)

        for trace in traces:
            if trace[0] != -1 and trace[1] != -1 and trace[2] != -1:
                start_feature = node_representations[start_node_type][trace[0].item()]
                mid_feature = node_representations[mid_node_type][trace[1].item()]
                end_feature = node_representations[end_node_type][trace[2].item()]
                concatenated_feature = torch.cat((start_feature, mid_feature, end_feature))
                # Save the meta-path type and features together
                meta_path_samples.append((path_type, concatenated_feature))

    return meta_path_samples



two_hop_paths = [
    ('cell', 'CT', 'target', 'TC', 'cell'),
    ('cell', 'CT', 'target', 'TD', 'drug'),
    ('cell', 'CT', 'target', 'TT1', 'target'),
    ('target', 'TC', 'cell', 'CC1', 'cell'),
    ('target', 'TD', 'drug', 'DT', 'target'),
    ('target', 'TT1', 'target', 'TC', 'cell'),
    ('target', 'TT1', 'target', 'TD', 'drug'),
    ('target', 'TT1', 'target', 'TT1', 'target'),
    ('drug', 'DT', 'target', 'TC', 'cell'),
    ('drug', 'DT', 'target', 'TD', 'drug'),
    ('drug', 'DT', 'target', 'TT1', 'target')]
