import argparse
import torch
import torch.nn.functional as F
### Argument and global variables
parser = argparse.ArgumentParser('dynamic contrastive anomaly detection')
parser.add_argument('-d', '--data', type=str,
                        help='Dataset name (eg. wikipedia / reddit / email-eu_testinj / digg_testinj / uci_testinj)',
                        default='wikipedia')
parser.add_argument('--bs', type=int, default=100, help='Batch_size')
parser.add_argument('--n_neighbors', type=int, default=20, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=30, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-6, help='Learning rate')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--num_layer', type=int, default=1, help='Number of layer')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')

parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=256, help='Dimensions of the msampleemory for '
                                                                    'each user')
parser.add_argument('--agg_type', type=str, default='TGAT', help='Aggregation type for memory recovery (only TGAT)')
parser.add_argument('--negative_memory_type', type=str, default='train', help='Negative memory type: train/random')
parser.add_argument('--message_updater', type=str, default='mlp', help='message updater type: mlp/identity')
parser.add_argument('--memory_updater', type=str, default="egru", choices=["gru", "rnn", "egru"],
                        help='Type of memory updater')
parser.add_argument('--training_ratio', type=float, default=0.85, help='training data ratio')
parser.add_argument('--lr_decay', type=float, default=0.7, help='learning rate decay ratio')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay ratio')

parser.add_argument('--srf', type=float, default=0.1,
                        help='weight of source memory reconstruction contrastive loss')
parser.add_argument('--drf', type=float, default=0.1,
                        help='weight of destination memory reconstruction contrastive loss')

parser.add_argument('--only_drift_loss_score', action='store_true', help='using drift loss and score only')
parser.add_argument('--only_recovery_loss_score', action='store_true', help='using recovery loss and score only')
parser.add_argument('--only_drift_score', action='store_true', help='using drift score and both loss')
parser.add_argument('--only_rec_score', action='store_true', help='using recovery score and both loss')
parser.add_argument('--test_inference_time', action='store_true', help='Test with inference time')

args = parser.parse_args()


def cosine_similarity(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

#wikipedia scales=[1.0, 0.5, 0.25]
def en_cos_sim(z1, z2, scales=[1, 1, 1]):
    result = 0
    for scale in scales:
        # 对特征进行不同尺度的池化或采样
        z1_scaled = F.avg_pool1d(z1.unsqueeze(1), kernel_size=int(1 / scale)).squeeze(1)
        z2_scaled = F.avg_pool1d(z2.unsqueeze(1), kernel_size=int(1 / scale)).squeeze(1)
        result += cosine_similarity(z1_scaled, z2_scaled)
    return result / len(scales)

class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)

