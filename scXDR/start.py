# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:29:16 2024

@author: 78760
"""




from training import *

S1, S2 ='GSE108383', 'GSE117872'

epochs = 2

train_model(S1, S2, epochs)



### 替代 ###
S1, S2 ='GSE117872', 'GSE127298'
epochs = 5
train_model(S1, S2, epochs)


from training_new import *          ### 5次中最佳结果

S1, S2 ='GSE164614', 'GSE230538_GSM7226481_pos_GSM7226483_neg'
epochs = 5
train_model(S1, S2, epochs)

S1, S2 ='GSE230538_GSM7226481_pos_GSM7226483_neg', 'GSE164614'
epochs = 5
train_model(S1, S2, epochs)


from training_new_temp import *      #### AUC > 1 才早停

S1, S2 ='GSE164614', 'GSE108394'
epochs = 6
train_model(S1, S2, epochs)


from training_feat_xiao import *

combinations = [
    ('GSE164614', 'GSE230538_GSM7226481_pos_GSM7226483_neg'),
    ('GSE230538_GSM7226481_pos_GSM7226483_neg', 'GSE164614')]

epochs = 3  # 设置 epochs
# 按指定组合逐一计算
for S1, S2 in combinations:
    print(f"Training model with S1={S1}, S2={S2}")
    train_model(S1, S2, epochs)


from training_stru_xiao import *

combinations = [
    ('GSE164614', 'GSE108394'),
    ('GSE164614', 'GSE230538_GSM7226481_pos_GSM7226483_neg'),
    ('GSE230538_GSM7226481_pos_GSM7226483_neg', 'GSE164614')]

epochs = 3  # 设置 epochs
# 按指定组合逐一计算
for S1, S2 in combinations:
    print(f"Training model with S1={S1}, S2={S2}")
    train_model(S1, S2, epochs)


from training_stru_xiao import *

S1, S2 ='GSE230538_GSM7226481_pos_GSM7226483_neg', 'GSE164614'
epochs = 3
train_model(S1, S2, epochs)













base_path = 'C:/Users/78760/Desktop/SCDS/Model/Graph'
tables_dir = "C:/Users/78760/Desktop/MyCode/scADR/试运行/表格"  # 存储表格文件的目录
models_dir = "C:/Users/78760/Desktop/MyCode/scADR/试运行/参数"  # 存储模型参数的目录

# 初始化表格，记录每个 epoch 的结果
source_results_df = pd.DataFrame(columns=['Epoch', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'F1 Score', 'Optimal Threshold'])
target_results_df = pd.DataFrame(columns=['Epoch', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'F1 Score', 'Optimal Threshold'])
all_pred_true_dfs = []


source_basic_graph, source_pos_graph, source_neg_graph, target_basic_graph, target_pos_graph, target_neg_graph, source_features, target_features = load_and_prepare_graphs(S1, S2, base_path)
source_features = {node_type: safe_min_max_normalize(features) for node_type, features in source_features.items()}
target_features = {node_type: safe_min_max_normalize(features) for node_type, features in target_features.items()}

prepare_seed()
folds = prepare_data_fold(source_pos_graph, source_neg_graph)
gcn_model, feature_aligners, edge_aligners, structure_aligners, link_predictor, predictor, optimizer, criterion, criterion_edge, max_edges_per_type = prepare_model()

# 训练模型
epochs = 5
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

        # Step 1: 特征对齐，更新节点特征并存储在新特征字典
        new_source_features = {}
        new_target_features = {}
    
        # 计算节点表示
        source_node_representations = gcn_model(source_basic_graph, source_features)
        target_node_representations = gcn_model(target_basic_graph, target_features)
    
        # 遍历节点类型，计算特征对齐
        for node_type in source_features.keys():
            src_encoded, src_decoded = feature_aligners(source_node_representations[node_type])
            tgt_encoded, tgt_decoded = feature_aligners(target_node_representations[node_type])
        
            new_source_features[node_type] = src_encoded  # 新特征
            new_target_features[node_type] = tgt_encoded
        
            # 计算重构损失
            loss_src = criterion(src_decoded, source_node_representations[node_type])
            loss_tgt = criterion(tgt_decoded, target_node_representations[node_type])
            feature_align_loss += (loss_src + loss_tgt)
        
            # 计算节点高密度点的对齐损失
            src_high_density_points = get_high_density_points(src_encoded, num_points=10)
            tgt_high_density_points = get_high_density_points(tgt_encoded, num_points=10)
        
            # 处理源域和目标域高密度点数量不一致的情况
            min_length = min(src_high_density_points.size(0), tgt_high_density_points.size(0))
            if min_length > 0:
                node_alignment_loss = calculate_alignment_loss(
                    src_high_density_points[:min_length],
                    tgt_high_density_points[:min_length])
                feature_align_loss += node_alignment_loss
            
        # total_loss += feature_align_loss*0.1
    
            # 计算每种边类型的对齐
        for etype in source_basic_graph.canonical_etypes:
            src_node_type, edge_type, tgt_node_type = etype
        
            # 获取源域的边索引
            src_edges = source_basic_graph.edges(etype=etype)
            num_src_edges = src_edges[0].size(0)
        
            # 随机选择边索引
            if num_src_edges > max_edges_per_type:
                random_src_indices = torch.randperm(num_src_edges)[:max_edges_per_type]
            else:
                random_src_indices = torch.arange(num_src_edges)
        
            # 获取源域的边特征
            source_edge_feats = torch.cat([
                source_node_representations[src_node_type][src_edges[0][random_src_indices]],
                source_node_representations[tgt_node_type][src_edges[1][random_src_indices]]
            ], dim=1)
        
            # 获取目标域的边索引
            tgt_edges = target_basic_graph.edges(etype=etype)
            num_tgt_edges = tgt_edges[0].size(0)
        
            # 随机选择边索引
            if num_tgt_edges > max_edges_per_type:
                random_tgt_indices = torch.randperm(num_tgt_edges)[:max_edges_per_type]
            else:
                random_tgt_indices = torch.arange(num_tgt_edges)
        
            # 获取目标域的边特征
            target_edge_feats = torch.cat([
                target_node_representations[src_node_type][tgt_edges[0][random_tgt_indices]],
                target_node_representations[tgt_node_type][tgt_edges[1][random_tgt_indices]]
            ], dim=1)
        
            # 边对齐自编码器
            src_edge_encoded, src_edge_decoded = edge_aligners(source_edge_feats)
            tgt_edge_encoded, tgt_edge_decoded = edge_aligners(target_edge_feats)
        
            # 计算边的重构损失
            loss_src_edge = criterion(src_edge_decoded, source_edge_feats)
            loss_tgt_edge = criterion(tgt_edge_decoded, target_edge_feats)
            feature_align_loss += (loss_src_edge + loss_tgt_edge)
        
            src_high_density_points = get_high_density_points(src_encoded, num_points=10)
            tgt_high_density_points = get_high_density_points(tgt_encoded, num_points=10)
        
            # 处理源域和目标域高密度点数量不一致的情况
            min_length = min(src_high_density_points.size(0), tgt_high_density_points.size(0))
            if min_length > 0:
                edge_alignment_loss = calculate_alignment_loss(
                    src_high_density_points[:min_length],
                    tgt_high_density_points[:min_length])
                feature_align_loss += edge_alignment_loss
        
        total_loss += feature_align_loss*0.2
    
        # 获取源域和目标域的元路径样本
        source_meta_paths = sample_meta_paths(source_basic_graph, new_source_features, two_hop_paths)
        target_meta_paths = sample_meta_paths(target_basic_graph, new_target_features, two_hop_paths)
        
        # 获取源域和目标域的元路径类型交集
        common_path_types = get_common_path_types(source_meta_paths, target_meta_paths)
        
        # 使用交集生成源域和目标域的元路径样本，确保类型一致
        source_common_meta_paths = sample_meta_paths(source_basic_graph, new_source_features, two_hop_paths, common_path_types)
        target_common_meta_paths = sample_meta_paths(target_basic_graph, new_target_features, two_hop_paths, common_path_types)
        
        # 遍历交集中的元路径类型，计算结构对齐的编码解码损失和对齐损失
        for path_type in common_path_types:
            # 获取该类型的源域和目标域元路径样本
            source_path_samples = [sample for sample in source_common_meta_paths if sample[0] == path_type]
            target_path_samples = [sample for sample in target_common_meta_paths if sample[0] == path_type]
            
            # 检查样本是否为空
            if not source_path_samples:
                print(f"No samples found for source path type: {path_type}")
                continue  # 跳过此元路径类型
            
            if not target_path_samples:
                print(f"No samples found for target path type: {path_type}")
                continue  # 跳过此元路径类型
        
            # 将源域和目标域元路径样本的特征提取到张量中
            source_feats = torch.stack([sample[1] for sample in source_path_samples])
            target_feats = torch.stack([sample[1] for sample in target_path_samples])
        
            # 使用结构对齐自编码器对每种类型的元路径特征进行编码和解码
            src_encoded, src_decoded = structure_aligners(source_feats)  # 假设元路径类型与起始节点类型相关
            tgt_encoded, tgt_decoded = structure_aligners(target_feats)
        
            # 计算重构损失
            loss_src = criterion(src_decoded, source_feats)
            loss_tgt = criterion(tgt_decoded, target_feats)
            structure_align_loss += (loss_src + loss_tgt)
        
            # 计算高密度点的对齐损失
            src_high_density_points = get_high_density_points(src_encoded, num_points=10)
            tgt_high_density_points = get_high_density_points(tgt_encoded, num_points=10)
        
            # 处理源域和目标域高密度点数量不一致的情况
            min_length = min(src_high_density_points.size(0), tgt_high_density_points.size(0))
            if min_length > 0:
                alignment_loss = calculate_alignment_loss(
                    src_high_density_points[:min_length],
                    tgt_high_density_points[:min_length])
                structure_align_loss += alignment_loss
    
        # 累加到总损失
        total_loss += structure_align_loss*0.2
        
    
        # 重建损失
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
    
        # 分几折
        test_mask, train_mask = Fold(folds, fold)
        
        # # Step 3: 使用对齐后的新特征字典进行边预测
        pos_score = predictor(source_pos_graph, new_source_features, ('drug', 'DCpos', 'cell'))
        neg_score = predictor(source_neg_graph, new_source_features, ('drug', 'DCneg', 'cell'))
        
        pos_score = pos_score[('drug', 'DCpos', 'cell')].squeeze()
        neg_score = neg_score[('drug', 'DCneg', 'cell')].squeeze()
    
        # 标签
        pos_labels = torch.ones(pos_score.shape[0], device=pos_score.device)
        neg_labels = torch.zeros(neg_score.shape[0], device=neg_score.device)
    
        # 拼接正负样本和标签
        scores = torch.cat([pos_score, neg_score], dim=0)[train_mask]
        labels = torch.cat([pos_labels, neg_labels], dim=0)[train_mask]
        
        edge_prediction_loss = criterion_edge(scores, labels)
        total_loss += edge_prediction_loss*5
        
        # 梯度清零、反向传播和参数更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 切换到评估模式
        predictor.eval()
        
        # 计算 ROC 曲线和最佳阈值
        with torch.no_grad():  # 使用 no_grad() 防止计算图的构建，提高评估效率
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
        
            # 依据最佳阈值计算预测类别
            y_pred = (y_prob >= optimal_threshold).astype(int)
        
            # 计算指标
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
        print(f"Total Loss: {total_loss.item():.4f}\n")
        
        print(f"Training - AUC: {train_results['AUC']:.4f}, AUPR: {train_results['AUPR']:.4f}, "
                  f"Accuracy: {train_results['Accuracy']:.4f}, Precision: {train_results['Precision']:.4f}, "
                  f"F1 Score: {train_results['F1 Score']:.4f}, Optimal Threshold: {train_results['Optimal Threshold']:.4f}")
        

        # 打印损失值
        print(f"Epoch {epoch+1}/{epochs}, Fold {fold+1}/{5}")
        print(f"Feature Align Loss: {feature_align_loss.item():.4f}")
        print(f"Structure Align Loss: {structure_align_loss.item():.4f}")
        print(f"Link Reconstruction: {linkloss.item():.4f}")
        print(f"Edge Prediction Loss: {edge_prediction_loss.item():.4f}")
        print(f"Total Loss: {total_loss.item():.4f}\n")
        
        
             # 检查是否满足提前停止条件
        if train_results['AUC'] > 0.99 and train_results['AUPR'] > 0.99 and train_results['Accuracy'] > 0.99:
             print("AUC and AUPR both exceed 0.99. Stopping training.")
             break  # 退出当前 fold 循环
       # 检查是否需要退出整个训练过程

    # 每个 epoch 结束后，计算源域和目标域的测试集指标
    # 源域测试集评估
    auc, aupr, acc, precision, f1, optimal_threshold, y_prob, y_true = evaluate_model(predictor, new_source_features, source_pos_graph, source_neg_graph, ('drug', 'DCpos', 'cell'))
    source_results_df.loc[epoch] = [epoch + 1, auc, aupr, acc, precision, f1, optimal_threshold]

    # 目标域测试集评估
    auc, aupr, acc, precision, f1, optimal_threshold, y_prob, y_true = evaluate_model(predictor, new_target_features, target_pos_graph, target_neg_graph, ('drug', 'DCpos', 'cell'))
    target_results_df.loc[epoch] = [epoch + 1, auc, aupr, acc, precision, f1, optimal_threshold]
    
    pred_true_df = pd.DataFrame({f'y_pred_epoch{epoch+1}': y_prob, 
                                 f'y_true_epoch{epoch+1}': y_true})
    all_pred_true_dfs.append(pred_true_df)
    
    if epoch == epochs-1:
        # 绘制得分点状图
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_prob)), y_prob, 
                    c=y_true, label='Score')
        plt.colorbar(label='True Label')  # 添加颜色条，表示标签（0 或 1）
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Score')
        plt.title('Predicted Scores with True Labels')
        plt.show()

    # 打印评估结果
    print(f"Source Domain - Epoch {epoch+1}/{epochs}")
    print(f"AUC: {source_results_df.loc[epoch, 'AUC']:.4f}, AUPR: {source_results_df.loc[epoch, 'AUPR']:.4f}, "
          f"Accuracy: {source_results_df.loc[epoch, 'Accuracy']:.4f}, Precision: {source_results_df.loc[epoch, 'Precision']:.4f}, "
          f"F1 Score: {source_results_df.loc[epoch, 'F1 Score']:.4f}, Optimal Threshold: {source_results_df.loc[epoch, 'Optimal Threshold']:.4f}")

    print(f"Target Domain - Epoch {epoch+1}/{epochs}")
    print(f"AUC: {target_results_df.loc[epoch, 'AUC']:.4f}, AUPR: {target_results_df.loc[epoch, 'AUPR']:.4f}, "
          f"Accuracy: {target_results_df.loc[epoch, 'Accuracy']:.4f}, Precision: {target_results_df.loc[epoch, 'Precision']:.4f}, "
          f"F1 Score: {target_results_df.loc[epoch, 'F1 Score']:.4f}, Optimal Threshold: {target_results_df.loc[epoch, 'Optimal Threshold']:.4f}\n")
    
    
    if train_results['AUC'] > 0.99 and train_results['AUPR'] > 0.99 and train_results['Accuracy'] > 0.99:
        break  # 退出所有循环

# 保存表格文件，包含 S1 和 S2 的标识符
source_results_df.to_csv(f"{tables_dir}/source_results_{S1}_{S2}.csv", index=False)
target_results_df.to_csv(f"{tables_dir}/target_results_{S1}_{S2}.csv", index=False)

# 拼接所有的 DataFrame 并保存
final_pred_true_df = pd.concat(all_pred_true_dfs, axis=1)
final_pred_true_df.to_csv(f"{tables_dir}/all_epochs_predictions_{S1}_{S2}.csv", index=False)

# 保存模型参数，包含 S1 和 S2 的标识符
torch.save(gcn_model.state_dict(), f"{models_dir}/gcn_model_{S1}_{S2}.pth")
torch.save(feature_aligners.state_dict(), f"{models_dir}/feature_aligners_{S1}_{S2}.pth")
torch.save(edge_aligners.state_dict(), f"{models_dir}/edge_aligners_{S1}_{S2}.pth")
torch.save(structure_aligners.state_dict(), f"{models_dir}/structure_aligners_{S1}_{S2}.pth")
torch.save(predictor.state_dict(), f"{models_dir}/predictor_{S1}_{S2}.pth")
torch.save(optimizer.state_dict(), f"{models_dir}/optimizer_{S1}_{S2}.pth")  # 可选：保存优化器状态

print("完成训练和测试。")


