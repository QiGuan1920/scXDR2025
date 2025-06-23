from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, roc_curve
import pandas as pd
import numpy as np

# Define the evaluation function
def evaluate_model(predictor, features, pos_graph, neg_graph):
    # Get the scores for positive edges
    pos_score = predictor(pos_graph, features, ('drug', 'DCpos', 'cell'))[('drug', 'DCpos', 'cell')].squeeze().detach().numpy()
    # Get the scores for negative edges
    neg_score = predictor(neg_graph, features, ('drug', 'DCneg', 'cell'))[('drug', 'DCneg', 'cell')].squeeze().detach().numpy()
    
    # Concatenate positive and negative scores
    y_prob = np.concatenate([pos_score, neg_score], axis=0)
    # Create the true labels (1 for positive edges, 0 for negative edges)
    y_true = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))], axis=0)
    
    # Calculate the false positive rate (fpr) and true positive rate (tpr) for ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    # Find the optimal threshold based on the maximum distance between tpr and fpr
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predict using the optimal threshold
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Calculate various evaluation metrics
    auc = roc_auc_score(y_true, y_prob)  # Area Under the ROC Curve
    aupr = average_precision_score(y_true, y_prob)  # Area Under the Precision-Recall Curve
    acc = accuracy_score(y_true, y_pred)  # Accuracy
    precision = precision_score(y_true, y_pred)  # Precision
    f1 = f1_score(y_true, y_pred)  # F1 Score
    
    # Return all evaluation metrics
    return auc, aupr, acc, precision, f1, optimal_threshold, y_prob, y_true
