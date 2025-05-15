import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import pdb
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def eval_anomaly_node_detection(model, data, batch_size, n_neighbors, device,
                                only_rec_score = False, only_drift_score=False, test_inference_time=False):
    pred_score = np.zeros(len(data.sources))
    pred_mask = np.zeros(len(data.sources), dtype=bool)
    timestamps_batch_all = []
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        model.eval()

        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            if test_inference_time:
                for idx in range(s_idx, e_idx):
                    model.neighbor_finder.node_to_neighbors[data.sources[idx]] = np.append(model.neighbor_finder.node_to_neighbors[data.sources[idx]], data.destinations[idx])
                    model.neighbor_finder.node_to_neighbors[data.destinations[idx]] = np.append(model.neighbor_finder.node_to_neighbors[data.destinations[idx]], data.sources[idx])
                    model.neighbor_finder.node_to_edge_timestamps[data.sources[idx]] = np.append(model.neighbor_finder.node_to_edge_timestamps[data.sources[idx]], data.timestamps[idx])
                    model.neighbor_finder.node_to_edge_timestamps[data.destinations[idx]] = np.append(model.neighbor_finder.node_to_edge_timestamps[data.destinations[idx]], data.timestamps[idx])

            sources_batch = torch.from_numpy(data.sources[s_idx: e_idx]).long().to(device)
            destinations_batch = torch.from_numpy(data.destinations[s_idx: e_idx]).long().to(device)
            timestamps_batch = torch.from_numpy(data.timestamps[s_idx:e_idx]).float().to(device)

            timestamps_batch_all.extend(timestamps_batch.cpu().numpy().tolist())

            src_neighbors_batch_np, _, src_neighbors_time_batch_np = model.neighbor_finder.get_temporal_neighbor(data.sources[s_idx: e_idx], data.timestamps[s_idx:e_idx], n_neighbors)
            dst_neighbors_batch_np, _, dst_neighbors_time_batch_np = model.neighbor_finder.get_temporal_neighbor(data.destinations[s_idx: e_idx], data.timestamps[s_idx:e_idx], n_neighbors)
            src_neighbors_batch = torch.from_numpy(src_neighbors_batch_np).long().to(device)
            dst_neighbors_batch = torch.from_numpy(dst_neighbors_batch_np).long().to(device)
            src_neighbors_time_batch = torch.from_numpy(src_neighbors_time_batch_np).long().to(device)
            dst_neighbors_time_batch = torch.from_numpy(dst_neighbors_time_batch_np).long().to(device)


            positive_memory_score, drift_score, _, _ = model.compute_anomaly_score(sources_batch, destinations_batch,
                                                                            timestamps_batch, src_neighbors_batch, dst_neighbors_batch,
                                                                            src_neighbors_time_batch, dst_neighbors_time_batch, n_neighbors)


            if np.isnan(positive_memory_score.reshape(-1).cpu().numpy()).sum()>0:
                pdb.set_trace()

            if only_drift_score: # ablation study
                pred_score[s_idx: e_idx] = (-(drift_score).reshape(-1).cpu().numpy() + 1)/2
            elif only_rec_score: # ablation study
                pred_score[s_idx: e_idx] = (-(positive_memory_score).reshape(-1).cpu().numpy() + 1)/2
            else: # default
                pred_score[s_idx: e_idx] = (-(drift_score).reshape(-1).cpu().numpy()-(positive_memory_score).reshape(-1).cpu().numpy() + 2)/4 #结合了时序漂移和重建误差两个维度来综合评估异常程度，分数越高表示越可能是异常样本。

        auc_roc = roc_auc_score(data.labels, pred_score)

        # plt.scatter(timestamps_batch_all, pred_score, alpha=0.7, color='blue', edgecolor='k')
        # plt.show()

        return auc_roc, pred_score, _, timestamps_batch_all