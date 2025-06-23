import torch
from models import *
from data_processing import *
from model_parameters import *
from evaluate import *
import pandas as pd
import numpy as np
import os
import time


def train_model(S1, S2, epochs):
    
    # Record start time
    start_time = time.time()
    start_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    
    # Set the main directory path for the results
    results_dir = "C:/Users/XXX/Desktop/scXDR/Trial_Run/Results"
    S1_S2_dir = f"{results_dir}/{S1}_{S2}"
    # Create S1_S2 folder and its subfolders: tables and models
    tables_subdir = f"{S1_S2_dir}/tables"
    models_subdir = f"{S1_S2_dir}/models"
    os.makedirs(tables_subdir, exist_ok=True)
    os.makedirs(models_subdir, exist_ok=True)
    
    base_path = 'C:/Users/XXX/Desktop/scXDR/Model/Graph'


    # Initialize tables to record results for each epoch
    source_results_df = pd.DataFrame(columns=['Epoch', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'F1 Score', 'Optimal Threshold'])
    target_results_df = pd.DataFrame(columns=['Epoch', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'F1 Score', 'Optimal Threshold'])
    all_pred_true_dfs = []

    
    source_basic_graph, source_pos_graph, source_neg_graph, target_basic_graph, target_pos_graph, target_neg_graph, source_features, target_features = load_and_prepare_graphs(S1, S2, base_path)
    source_features = {node_type: safe_min_max_normalize(features) for node_type, features in source_features.items()}
    target_features = {node_type: safe_min_max_normalize(features) for node_type, features in target_features.items()}
    
    prepare_seed()
    folds = prepare_data_fold(source_pos_graph, source_neg_graph)
    gcn_model, feature_aligners, edge_aligners, structure_aligners, link_predictor, predictor, optimizer, criterion, criterion_edge, max_edges_per_type = prepare_model()
    
    # Train the model
    for epoch in range(epochs):
        for fold in range(5):

            gcn_model.train()
            feature_aligners.train()
            edge_aligners.train()
            structure_aligners.train()
            predictor.train()
    
            total_loss = 0
            feature_align_loss = 0
            structure_align_loss = 0
    
            # Feature alignment, update node features and store in new feature dictionary
            new_source_features = {}
            new_target_features = {}
        
            # Compute node representations
            source_node_representations = gcn_model(source_basic_graph, source_features)
            target_node_representations = gcn_model(target_basic_graph, target_features)
        
            # Iterate over node types to compute feature alignment
            for node_type in source_features.keys():
                src_encoded, src_decoded = feature_aligners(source_node_representations[node_type])
                tgt_encoded, tgt_decoded = feature_aligners(target_node_representations[node_type])
            
                new_source_features[node_type] = src_encoded  # New feature
                new_target_features[node_type] = tgt_encoded
            
                # Compute reconstruction loss
                loss_src = criterion(src_decoded, source_node_representations[node_type])
                loss_tgt = criterion(tgt_decoded, target_node_representations[node_type])
                feature_align_loss += (loss_src + loss_tgt)
            
                # Compute alignment loss for high-density nodes
                src_high_density_points = get_high_density_points(src_encoded, num_points=10)
                tgt_high_density_points = get_high_density_points(tgt_encoded, num_points=10)
            
                # Handle case where the number of high-density points is different between source and target domains
                min_length = min(src_high_density_points.size(0), tgt_high_density_points.size(0))
                if min_length > 0:
                    node_alignment_loss = calculate_alignment_loss(
                        src_high_density_points[:min_length],
                        tgt_high_density_points[:min_length])
                    feature_align_loss += node_alignment_loss
                
            # total_loss += feature_align_loss*0.1
        
            # Compute alignment for each edge type
            for etype in source_basic_graph.canonical_etypes:
                src_node_type, edge_type, tgt_node_type = etype
            
                # Get source domain edge indices
                src_edges = source_basic_graph.edges(etype=etype)
                num_src_edges = src_edges[0].size(0)
            
                # Randomly select edge indices
                if num_src_edges > max_edges_per_type:
                    random_src_indices = torch.randperm(num_src_edges)[:max_edges_per_type]
                else:
                    random_src_indices = torch.arange(num_src_edges)
            
                # Get source domain edge features
                source_edge_feats = torch.cat([
                    source_node_representations[src_node_type][src_edges[0][random_src_indices]],
                    source_node_representations[tgt_node_type][src_edges[1][random_src_indices]]
                ], dim=1)
            
                # Get target domain edge indices
                tgt_edges = target_basic_graph.edges(etype=etype)
                num_tgt_edges = tgt_edges[0].size(0)
            
                # Randomly select edge indices
                if num_tgt_edges > max_edges_per_type:
                    random_tgt_indices = torch.randperm(num_tgt_edges)[:max_edges_per_type]
                else:
                    random_tgt_indices = torch.arange(num_tgt_edges)
            
                # Get target domain edge features
                target_edge_feats = torch.cat([
                    target_node_representations[src_node_type][tgt_edges[0][random_tgt_indices]],
                    target_node_representations[tgt_node_type][tgt_edges[1][random_tgt_indices]]
                ], dim=1)
            
                # Edge alignment autoencoder
                src_edge_encoded, src_edge_decoded = edge_aligners(source_edge_feats)
                tgt_edge_encoded, tgt_edge_decoded = edge_aligners(target_edge_feats)
            
                # Compute edge reconstruction loss
                loss_src_edge = criterion(src_edge_decoded, source_edge_feats)
                loss_tgt_edge = criterion(tgt_edge_decoded, target_edge_feats)
                feature_align_loss += (loss_src_edge + loss_tgt_edge)
            
                src_high_density_points = get_high_density_points(src_encoded, num_points=10)
                tgt_high_density_points = get_high_density_points(tgt_encoded, num_points=10)
            
                # Handle case where the number of high-density points is different between source and target domains
                min_length = min(src_high_density_points.size(0), tgt_high_density_points.size(0))
                if min_length > 0:
                    edge_alignment_loss = calculate_alignment_loss(
                        src_high_density_points[:min_length],
                        tgt_high_density_points[:min_length])
                    feature_align_loss += edge_alignment_loss
            
            total_loss += feature_align_loss*0.2
        
            # Get meta-path samples for source and target domains
            source_meta_paths = sample_meta_paths(source_basic_graph, new_source_features, two_hop_paths)
            target_meta_paths = sample_meta_paths(target_basic_graph, new_target_features, two_hop_paths)
            
            # Get the intersection of meta-path types for source and target domains
            common_path_types = get_common_path_types(source_meta_paths, target_meta_paths)
            
            # Use intersection to generate meta-path samples for source and target domains, ensuring type consistency
            source_common_meta_paths = sample_meta_paths(source_basic_graph, new_source_features, two_hop_paths, common_path_types)
            target_common_meta_paths = sample_meta_paths(target_basic_graph, new_target_features, two_hop_paths, common_path_types)
            
            # Iterate over the intersection of meta-path types, compute structural alignment encoding-decoding losses and alignment losses
            for path_type in common_path_types:
                # Get source and target domain meta-path samples for this type
                source_path_samples = [sample for sample in source_common_meta_paths if sample[0] == path_type]
                target_path_samples = [sample for sample in target_common_meta_paths if sample[0] == path_type]
                
                # Skip if no samples found
                if not source_path_samples:
                    print(f"No samples found for source path type: {path_type}")
                    continue  # Skip this meta-path type
                
                if not target_path_samples:
                    print(f"No samples found for target path type: {path_type}")
                    continue  # Skip this meta-path type
            
                # Extract features from the source and target domain meta-path samples
                source_feats = torch.stack([sample[1] for sample in source_path_samples])
                target_feats = torch.stack([sample[1] for sample in target_path_samples])
            
                # Use structure alignment autoencoder for encoding and decoding features of each meta-path type
                src_encoded, src_decoded = structure_aligners(source_feats)
                tgt_encoded, tgt_decoded = structure_aligners(target_feats)
            
                # Compute reconstruction loss
                loss_src = criterion(src_decoded, source_feats)
                loss_tgt = criterion(tgt_decoded, target_feats)
                structure_align_loss += (loss_src + loss_tgt)
            
                # Compute alignment loss for high-density points
                src_high_density_points = get_high_density_points(src_encoded, num_points=10)
                tgt_high_density_points = get_high_density_points(tgt_encoded, num_points=10)
            
                # Handle case where the number of high-density points is different between source and target domains
                min_length = min(src_high_density_points.size(0), tgt_high_density_points.size(0))
                if min_length > 0:
                    alignment_loss = calculate_alignment_loss(
                        src_high_density_points[:min_length],
                        tgt_high_density_points[:min_length])
                    structure_align_loss += alignment_loss
        
            # Accumulate to total loss
            total_loss += structure_align_loss*0.2
            
        
            # Reconstruction loss
            source_loss1 = lossloss(link_predictor, source_basic_graph, 1, new_source_features, etype=('drug', 'DT', 'target'))
            source_loss2 = lossloss(link_predictor, source_basic_graph, 1, new_source_features, etype=('cell', 'CT', 'target'))
            source_loss3 = lossloss(link_predictor, source_basic_graph, 1, new_source_features, etype=('cell', 'CC1', 'cell'))
            source_loss4 = lossloss(link_predictor, source_basic_graph, 1, new_source_features, etype=('target', 'TT1', 'target'))
            target_loss1 = lossloss(link_predictor, target_basic_graph, 1, new_target_features, etype=('drug', 'DT', 'target'))
            target_loss2 = lossloss(link_predictor, target_basic_graph, 1, new_target_features, etype=('cell', 'CT', 'target'))
            target_loss3 = lossloss(link_predictor, target_basic_graph, 1, new_target_features, etype=('cell', 'CC1', 'cell'))
            target_loss4 = lossloss(link_predictor, target_basic_graph, 1, new_target_features, etype=('target', 'TT1', 'target'))
            
            linkloss = source_loss1+source_loss2+source_loss3+source_loss4+target_loss1+target_loss2+target_loss3+target_loss4
            
            total_loss += linkloss
        
            # Cross-validation
            test_mask, train_mask = Fold(folds, fold)
            
            # # Step 3: Use the aligned new feature dictionary for edge prediction
            pos_score = predictor(source_pos_graph, new_source_features, ('drug', 'DCpos', 'cell'))
            neg_score = predictor(source_neg_graph, new_source_features, ('drug', 'DCneg', 'cell'))
            
            pos_score = pos_score[('drug', 'DCpos', 'cell')].squeeze()
            neg_score = neg_score[('drug', 'DCneg', 'cell')].squeeze()
        
            # Labels
            pos_labels = torch.ones(pos_score.shape[0], device=pos_score.device)
            neg_labels = torch.zeros(neg_score.shape[0], device=neg_score.device)
        
            # Concatenate positive and negative samples and labels
            scores = torch.cat([pos_score, neg_score], dim=0)[train_mask]
            labels = torch.cat([pos_labels, neg_labels], dim=0)[train_mask]
            
            edge_prediction_loss = criterion_edge(scores, labels)
            total_loss += edge_prediction_loss*5
            
            # Zero gradients, backpropagate, and update parameters
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Switch to evaluation mode
            predictor.eval()
            
            # Compute ROC curve and optimal threshold
            with torch.no_grad():  # Use no_grad() to prevent building the computation graph, improving evaluation efficiency
                pos_score = predictor(source_pos_graph, new_source_features, ('drug', 'DCpos', 'cell'))
                neg_score = predictor(source_neg_graph, new_source_features, ('drug', 'DCneg', 'cell'))
            
                pos_score = pos_score[('drug', 'DCpos', 'cell')].squeeze()
                neg_score = neg_score[('drug', 'DCneg', 'cell')].squeeze()
            
                scores = torch.cat([pos_score, neg_score], dim=0)[test_mask]
                labels = torch.cat([pos_labels, neg_labels], dim=0)[test_mask]
            
                y_true = labels.detach().numpy()
                y_prob = scores.detach().numpy()
            
                fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                j_index = tpr - fpr
                optimal_idx = np.argmax(j_index)
                optimal_threshold = thresholds[optimal_idx]
            
                # Predict using the optimal threshold
                y_pred = (y_prob >= optimal_threshold).astype(int)
            
                # Compute metrics
                auc = roc_auc_score(y_true, y_prob)
                aupr = average_precision_score(y_true, y_prob)
                acc = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            
                train_results = {
                    'AUC': auc,
                    'AUPR': aupr,
                    'Accuracy': acc,
                    'Precision': precision,
                    'F1 Score': f1,
                    'Optimal Threshold': optimal_threshold,
                    'y_true': y_true,
                    'y_prob': y_prob
                }
            
        
            print(f"Epoch {epoch+1}/{epochs}", f"Fold {fold+1}/{5}")
            print(f"Node Align Loss: {node_alignment_loss.item():.4f}")
            # print(f"Edge Align Loss: {edge_alignment_loss.item():.4f}")
            print(f"Feature Align Loss: {feature_align_loss.item():.4f}")
            print(f"Structure Align Loss: {structure_align_loss.item():.4f}")
            print(f"Link Reconstruction: {linkloss.item():.4f}")
            print(f"Edge Prediction Loss: {edge_prediction_loss.item():.4f}")
            print(f"Total Loss: {total_loss.item():.4f}")
            
            print(f"Training - AUC: {train_results['AUC']:.4f}, AUPR: {train_results['AUPR']:.4f}, "
                      f"Accuracy: {train_results['Accuracy']:.4f}, Precision: {train_results['Precision']:.4f}, "
                      f"F1 Score: {train_results['F1 Score']:.4f}, Optimal Threshold: {train_results['Optimal Threshold']:.4f}\n")
    
                 # Check if early stopping criteria are met
            if train_results['AUC'] > 0.99 and train_results['AUPR'] > 0.99 and train_results['Accuracy'] > 0.99:
                 print("AUC and AUPR both exceed 0.99. Stopping fold training.\n")
                 break  # Exit the current fold loop
    
        # After each epoch, compute evaluation metrics for source and target domain test sets
        # Source domain test set evaluation
        auc, aupr, acc, precision, f1, optimal_threshold, y_prob, y_true = evaluate_model(predictor, new_source_features, source_pos_graph, source_neg_graph)
        source_results_df.loc[epoch] = [epoch + 1, auc, aupr, acc, precision, f1, optimal_threshold]
    
        # Target domain test set evaluation
        auc, aupr, acc, precision, f1, optimal_threshold, y_prob, y_true = evaluate_model(predictor, new_target_features, target_pos_graph, target_neg_graph)
        target_results_df.loc[epoch] = [epoch + 1, auc, aupr, acc, precision, f1, optimal_threshold]
        
        pred_true_df = pd.DataFrame({f'y_pred_epoch{epoch+1}': y_prob, 
                                     f'y_true_epoch{epoch+1}': y_true})
        all_pred_true_dfs.append(pred_true_df)  ### List format
        
        if epoch == epochs-1:
            # Plot the score scatter plot
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(y_prob)), y_prob, 
                        c=y_true, label='Score')
            plt.colorbar(label='True Label')  # Add color bar for labels (0 or 1)
            plt.xlabel('Sample Index')
            plt.ylabel('Predicted Score')
            plt.title('Predicted Scores with True Labels')
            plt.show()

        # Print evaluation results
        print(f"Source Domain - Epoch {epoch+1}/{epochs}")
        print(f"AUC: {source_results_df.loc[epoch, 'AUC']:.4f}, AUPR: {source_results_df.loc[epoch, 'AUPR']:.4f}, "
              f"Accuracy: {source_results_df.loc[epoch, 'Accuracy']:.4f}, Precision: {source_results_df.loc[epoch, 'Precision']:.4f}, "
              f"F1 Score: {source_results_df.loc[epoch, 'F1 Score']:.4f}, Optimal Threshold: {source_results_df.loc[epoch, 'Optimal Threshold']:.4f}")
    
        print(f"Target Domain - Epoch {epoch+1}/{epochs}")
        print(f"AUC: {target_results_df.loc[epoch, 'AUC']:.4f}, AUPR: {target_results_df.loc[epoch, 'AUPR']:.4f}, "
              f"Accuracy: {target_results_df.loc[epoch, 'Accuracy']:.4f}, Precision: {target_results_df.loc[epoch, 'Precision']:.4f}, "
              f"F1 Score: {target_results_df.loc[epoch, 'F1 Score']:.4f}, Optimal Threshold: {target_results_df.loc[epoch, 'Optimal Threshold']:.4f}\n")
        
        # Save the result tables, including S1 and S2 identifiers
        source_results_df.to_csv(f"{tables_subdir}/source_results_{S1}_{S2}.csv", index=False)
        target_results_df.to_csv(f"{tables_subdir}/target_results_{S1}_{S2}.csv", index=False)
        
        # Concatenate all DataFrames and save
        final_pred_true_df = pd.concat(all_pred_true_dfs, axis=1)
        final_pred_true_df.to_csv(f"{tables_subdir}/all_epochs_predictions_{S1}_{S2}.csv", index=False)
        
        
        if train_results['AUC'] > 0.99 and train_results['AUPR'] > 0.99 and train_results['Accuracy'] > 0.99:
            print("AUC and AUPR both exceed 0.99. Stopping epoch training.\n")
            break  # Exit all loops
    
    # After training and testing, record end time
    end_time = time.time()
    end_time_readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    
    # Calculate total runtime
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_minutes = int((elapsed_time % 3600) // 60)
    elapsed_seconds = int(elapsed_time % 60)
    
    # Output total runtime
    print(f"Training completed in {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s.\n")
    
    # Save start time, end time, and elapsed time (seconds and formatted) to the table folder
    with open(f"{tables_subdir}/training_time_{S1}_{S2}.txt", "w") as f:
        f.write(f"Training started at: {start_time_readable}\n")
        f.write(f"Training completed at: {end_time_readable}\n")
        f.write(f"Total time elapsed (formatted): {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s.\n")
        f.write(f"Total time elapsed (seconds): {elapsed_time:.2f} seconds\n")
    
    
    # Save model parameters, including S1 and S2 identifiers
    torch.save(gcn_model.state_dict(), f"{models_subdir}/gcn_model_{S1}_{S2}.pth")
    torch.save(feature_aligners.state_dict(), f"{models_subdir}/feature_aligners_{S1}_{S2}.pth")
    torch.save(edge_aligners.state_dict(), f"{models_subdir}/edge_aligners_{S1}_{S2}.pth")
    torch.save(structure_aligners.state_dict(), f"{models_subdir}/structure_aligners_{S1}_{S2}.pth")
    torch.save(predictor.state_dict(), f"{models_subdir}/predictor_{S1}_{S2}.pth")
    torch.save(optimizer.state_dict(), f"{models_subdir}/optimizer_{S1}_{S2}.pth")  # Optional: Save optimizer state
    
    print("Training and testing completed.")
