# encoders/egnn_encoder.py

from egnn_pytorch import EGNN, EGNN_Network, EGNN_Sparse
from torch_geometric.nn import global_max_pool
import torch.nn as nn
import torch.nn.functional as F


class EGNNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=128, num_layers=3):
        super(EGNNBlock, self).__init__()
        self.out_dim = out_channels
        self.node_embed = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                EGNN_Sparse(
                    feats_dim=hidden_channels,
                    m_dim=hidden_channels,
                    fourier_features=0 # Можно включить для лучшей работы с расстояниями
                )
            )
        
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, pos, edge_index, batch):
        # 1. Линейная проекция фичей
        x = self.node_embed(x)
        
        # 2. Прогон через эквивариантные слои
        for layer in self.layers:
            # EGNN_Sparse обновляет и фичи, и координаты!
            x, pos = layer(x, pos, edge_index, batch)
            
        # 3. Агрегация графа в единый вектор
        x = global_max_pool(x, batch)
        
        # 4. Финальная проекция и нормализация
        x = self.out_proj(x)
        return self.batch_norm(x)
