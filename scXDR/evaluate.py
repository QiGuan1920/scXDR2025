# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:37:28 2024

@author: 78760
"""

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, roc_curve
import pandas as pd
import numpy as np

# 定义评估函数
def evaluate_model(predictor, features, pos_graph, neg_graph):
    pos_score = predictor(pos_graph, features, ('drug', 'DCpos', 'cell'))[('drug', 'DCpos', 'cell')].squeeze().detach().numpy()
    neg_score = predictor(neg_graph, features, ('drug', 'DCneg', 'cell'))[('drug', 'DCneg', 'cell')].squeeze().detach().numpy()
    y_prob = np.concatenate([pos_score, neg_score], axis=0)
    y_true = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))], axis=0)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return auc, aupr, acc, precision, f1, optimal_threshold, y_prob, y_true