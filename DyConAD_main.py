from utils.utils import parser
import torch
import numpy as np
import random
from pathlib import Path
import logging
import time
from data_process.data_process import get_data_node_classification
from utils.neighor_finder import get_neighbor_finder
from model.TAD import TAD
import math
from tqdm import tqdm
import psutil
import os
from utils.evaluation import eval_anomaly_node_detection

if __name__ == '__main__':
    args = parser.parse_args()
    #----------------定义随机种子，使代码结果易复现------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #---------------------定义变量-------------------------
    DATA = args.data
    GPU = args.gpu
    BATCH_SIZE = args.bs
    NUM_EPOCH = args.n_epoch
    NUM_NEIGHBORS = args.n_neighbors
    NUM_HEADS = args.n_head #多头注意力头数
    DROP_OUT = args.drop_out
    NUM_LAYER = args.num_layer
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim
    memory_agg_type = args.agg_type
    negative_memory_type = args.negative_memory_type
    message_updater = args.message_updater
    #------------------创建一个log目录存储信息-----------------
    Path("log/").mkdir(parents=True, exist_ok=True)
    #-----------------记录程序执行过程中的信息----------------
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}srf{}_drf{}_epoch{}_lr{}_bs{}_memdim{}_msgdim{}_{}.log'.format(
        DATA, args.srf, args.drf, NUM_EPOCH, LEARNING_RATE, BATCH_SIZE, MEMORY_DIM, MESSAGE_DIM, str(time.time()))) #包含了实验的各种参数，便于区分不同实验的日志
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #定义日志格式，包括时间、名称、级别和消息
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    #----------------------数据处理----------------------
    logger.info(args) #记录命令行参数
    logger.info('Data Processing') #记录数据处理开始的信息
    full_data, train_data, test_data = \
        get_data_node_classification(DATA, training_ratio=args.training_ratio)
    #--------------确定最大值节点----------------
    max_idx = max(full_data.unique_nodes)
    #-------------创建 NeighborFinder 对象 [邻居节点，边，时间戳]---------
    train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)
    full_ngh_finder = get_neighbor_finder(full_data, uniform=False, max_node_idx=max_idx)
    #-------------邻居采样 右对齐，若无邻居填充0---------
    src_neighbors, _, src_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(train_data.sources, train_data.timestamps, NUM_NEIGHBORS)
    dst_neighbors, _, dst_neighbors_time = train_ngh_finder.get_temporal_neighbor_tqdm(train_data.destinations, train_data.timestamps, NUM_NEIGHBORS)
    #-------------------使用GPU或CPU------------------------
    device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_string)

    #------------------计算每一轮所用时间------------------------
    epoch_times = []
    #----------------汇总测试结果--------------------------
    test_aucs = []
    #----------------------------进行训练学习-----------------------------
    for i in range(args.n_runs):
        logger.info('Dynamic anomaly detection start - runs: {}'.format(str(i)))
        tad = TAD(neighbor_finder=train_ngh_finder, n_nodes=full_data.n_unique_nodes, n_edges=full_data.n_interactions,
                                 device=device,n_layers=NUM_LAYER, n_heads=NUM_HEADS,
                                dropout=DROP_OUT, message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM, n_neighbors=NUM_NEIGHBORS,
                                memory_agg_type=memory_agg_type, negative_memory_type=negative_memory_type, message_updater=message_updater,
                                memory_updater=args.memory_updater, src_reg_factor=args.srf, dst_reg_factor=args.drf,
                                only_drift_loss=args.only_drift_loss_score, only_recovery_loss = args.only_recovery_loss_score)

    tad = tad.to(device)
    train_data_sources = torch.from_numpy(train_data.sources).long().to(device)
    train_data_destinations = torch.from_numpy(train_data.destinations).long().to(device)
    train_data_timestamps = torch.from_numpy(train_data.timestamps).float().to(device)
    train_data_src_neighbors = torch.from_numpy(src_neighbors).long().to(device)
    train_data_dst_neighbors = torch.from_numpy(dst_neighbors).long().to(device)
    train_data_src_neighbors_time = torch.from_numpy(src_neighbors_time).long().to(device)
    train_data_dst_neighbors_time = torch.from_numpy(dst_neighbors_time).long().to(device)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    optimizer = torch.optim.Adam(tad.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    time_list = []
    train_time_list = []
    negative_train_nodes = torch.from_numpy(
        np.array(list(set(train_data.destinations) | set(train_data.sources)))).long().to(device)

    for epoch in range(NUM_EPOCH):
        tad.memory.__init_memory__()
        tad.set_neighbor_finder(train_ngh_finder)
        m_loss = []
        val_aucs = []

        train_start = time.time()
        for k in tqdm(range(num_batch)):
            loss = 0
            optimizer.zero_grad()
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance, s_idx + BATCH_SIZE)

            sources_batch = train_data_sources[s_idx: e_idx]
            destinations_batch = train_data_destinations[s_idx: e_idx]
            timestamps_batch = train_data_timestamps[s_idx: e_idx]
            src_neighbors_batch = train_data_src_neighbors[s_idx: e_idx]
            dst_neighbors_batch = train_data_dst_neighbors[s_idx: e_idx]
            src_neighbors_time_batch = train_data_src_neighbors_time[s_idx: e_idx]
            dst_neighbors_time_batch = train_data_dst_neighbors_time[s_idx: e_idx]

            size = len(sources_batch)

            tad.train()
            _, _, _, _, contrastive_loss = tad.compute_node_diff_score(sources_batch,
                                                                           destinations_batch,
                                                                           timestamps_batch,
                                                                           src_neighbors_batch,
                                                                           dst_neighbors_batch,
                                                                           src_neighbors_time_batch,
                                                                           dst_neighbors_time_batch,
                                                                           NUM_NEIGHBORS,
                                                                           negative_train_nodes)  # train_data.n_unique_nodes
            loss += contrastive_loss
            if args.only_drift_loss_score and k == 0:
                continue
            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())
            tad.memory.detach_memory()
        train_end = time.time()
        train_time_list.append(train_end - train_start)

        scheduler.step()
        if args.test_inference_time:
            train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)
            tad.set_neighbor_finder(train_ngh_finder)
        else:
            tad.set_neighbor_finder(full_ngh_finder)

        start = time.time()
        val_auc, pred_score, pred_mask, timestamps_batch_all = eval_anomaly_node_detection(tad, test_data,
                                                                                           BATCH_SIZE,
                                                                                           n_neighbors=NUM_NEIGHBORS,
                                                                                           device=device,
                                                                                           only_rec_score=args.only_rec_score or args.only_recovery_loss_score,
                                                                                           only_drift_score=args.only_drift_loss_score or args.only_drift_score,
                                                                                           test_inference_time=args.test_inference_time)

        end = time.time()
        time_list.append(end - start)
        logger.info("Epoch {} - mloss: {:.4f} val auc: {:.4f}".format(str(epoch), sum(m_loss) / len(m_loss), val_auc))
        print(u'当前进程的内存使用:%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))
        val_aucs.append(val_auc)










