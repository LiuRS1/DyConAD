import numpy as np
from tqdm import tqdm


def get_neighbor_finder(data, uniform=False, max_node_idx=None):
    max_node_idx = max(data.sources.max(), data.destinations.max()) if max_node_idx is None else max_node_idx #如果未提供 max_node_idx，则计算源节点和目标节点中的最大索引值
    adj_list = [[] for _ in range(max_node_idx + 1)] #确定邻接矩阵大小 创建一个长度为 max_node_idx + 1 的列表，每个元素是一个空列表，索引对应节点ID，列表中存储该节点的邻居信息
    for source, destination, edge_idx, timestamp in zip(data.sources, data.destinations,
                                                      data.edge_idxs,
                                                      data.timestamps):
        adj_list[source].append((destination, edge_idx, timestamp)) #遍历所有边（source → destination） 确保source中有destination []
        adj_list[destination].append((source, edge_idx, timestamp)) #确保destination中有source
    return NeighborFinder(adj_list, uniform=uniform) #使用构建的邻接列表创建 NeighborFinder 对象 [邻居节点，边，时间戳]


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []  # 存储每个节点的邻居节点ID
        self.node_to_edge_idxs = []  # 存储每个节点的边索引
        self.node_to_edge_timestamps = [] #存储每个节点的边时间戳
        self.uniform = uniform

#--------------对每个节点的邻居按时间戳排序，并分别存储邻居ID、边索引和时间戳-----------
        for neighbors in adj_list:
            sorted_neighhbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(np.array([x[0] for x in sorted_neighhbors]))
            self.node_to_edge_idxs.append(np.array([x[1] for x in sorted_neighhbors]))
            self.node_to_edge_timestamps.append(np.array([x[2] for x in sorted_neighhbors]))

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)
#------------使用二分查找（np.searchsorted）找到时间戳小于 cut_time 的最后一个位置 返回该节点在截止时间前的所有邻居节点、边索引和时间戳---------
    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[
                                                                                             src_idx][:i]
#---------一组源节点在指定时间点之前采样时序邻居------------------------
    def get_temporal_neighbor_tqdm(self, source_nodes, timestamps, n_neighbors=20):
        assert (len(source_nodes) == len(timestamps))
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in tqdm(enumerate(zip(source_nodes, timestamps))):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                                                     timestamp)  # extracts all neighbors, interactions indexes and timestamps of all interactions of user source_node happening before cut_time

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors) #从所有可能的邻居中随机均匀地选择 n_neighbors 个索引，每个索引被选中的概率是相等的

                    neighbors[i, :] = source_neighbors[sampled_idx] #随机选择的邻居节点ID进行赋值
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]#重新排列顺序
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    source_edge_times = source_edge_times[-n_neighbors:] #最近邻采样 从源数组中选取最后 n_neighbors 个元素
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors) #采样数不超过n_neighbors
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors  #填充位置 结果右对齐，如果邻居数量不足 n_neighbors，左侧会留空（默认填充0）
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        assert (len(source_nodes) == len(timestamps))
        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.float32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]

        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times = self.find_before(source_node,
                                                                                     timestamp)

            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]

                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                else:
                    source_edge_times = source_edge_times[-n_neighbors:]
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs

        return neighbors, edge_idxs, edge_times