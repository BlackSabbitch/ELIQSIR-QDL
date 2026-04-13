# encoders/gnn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool

class FlexGNNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, out_channels=128,
                 num_layers=3, conv_type='gcn', heads=4):
        """
        in_channels: Количество фич в каждом узле (например, 3 для координат [X, Y, Z])
        hidden_channels: Размер скрытого слоя графовой свертки
        out_channels: Размер итогового эмбеддинга (должен совпадать с embed_dim = 128)
        num_layers: Количество слоев графовой свертки (обычно 2-4)
        """
        super(FlexGNNBlock, self).__init__()

        self.out_dim = out_channels
        self.convs = nn.ModuleList()
        self.conv_type = conv_type.lower()
        
        if self.conv_type == 'gat':
            # GAT склеивает выходы от разных голов (concat=True по умолчанию),
            # поэтому размер скрытого слоя для каждой головы нужно поделить на их количество
            gat_hidden = hidden_channels // heads
            
            # Первый слой
            self.convs.append(GATConv(in_channels, gat_hidden, heads=heads))
            # Скрытые слои
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels, gat_hidden, heads=heads))
            # Последний слой: concat=False, чтобы усреднить головы и получить ровно out_channels
            self.convs.append(GATConv(hidden_channels, out_channels, heads=heads, concat=False))
            
        elif self.conv_type == 'gcn':
            # Классический GCN (как мы писали ранее)
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
            
        else:
            raise ValueError(f"Неизвестный тип свертки: {conv_type}. Используйте 'gcn' или 'gat'.")
            
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, batch):
        """
        x: Тензор фич узлов [num_nodes, in_channels]
        edge_index: Тензор связей [2, num_edges]
        batch: Вектор, указывающий, какому графу в батче принадлежит каждый узел
        """
        # Прогоняем через графовые свертки
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                # x = F.elu(x) if self.conv_type == 'gat' else F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
                
        # Глобальный пулинг: сжимаем весь граф (все узлы) в один вектор для каждого графа в батче
        # Используем global_max_pool (как в CNN статье)
        x = global_max_pool(x, batch)
        
        # Нормализация
        x = self.batch_norm(x)
        
        return x
