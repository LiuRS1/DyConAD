import numpy as np
import pandas as pd

def get_data_node_classification(dataset_name, training_ratio=0.85):
  ### 加载和处理时序图数据，并按照时间顺序将数据分割为训练集和测试集
  graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))

  test_time = np.quantile(graph_df.ts, training_ratio)
  sources = graph_df.u.values
  destinations = graph_df.i.values
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  train_mask = timestamps <= test_time #训练
  test_mask = timestamps > test_time #测试

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])#训练集 数据对象

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask]) #测试集

  return full_data, train_data, test_data


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)

