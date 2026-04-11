# model.py

import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp

# --- МОДУЛЬ 1: ПАРАЛЛЕЛЬНЫЕ CNN (Multi-scale) ---
class FlexCNNBlock(nn.Module):
    def __init__(self, vocab_size, embed_dim, mode='parallel', filters=32, kernels=[4, 8, 12]):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if mode == 'parallel':
            # Логика из статьи: независимые фильтры разного размера
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, filters, kernel_size=k, padding=k//2) 
                for k in kernels
            ])
            self.out_dim = filters * len(kernels)
        
        elif mode == 'sequential':
            # Классический глубокий подход: слои друг за другом
            # Каждый следующий слой «видит» комбинации признаков предыдущего
            self.net = nn.Sequential(
                nn.Conv1d(embed_dim, filters, kernel_size=kernels[0], padding=kernels[0]//2),
                nn.ReLU(),
                nn.Conv1d(filters, filters * 2, kernel_size=kernels[1], padding=kernels[1]//2),
                nn.ReLU(),
                nn.Conv1d(filters * 2, filters * 3, kernel_size=kernels[2], padding=kernels[2]//2),
                nn.ReLU()
            )
            self.out_dim = filters * 3 # 32 * 3 = 96
            
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        
        if self.mode == 'parallel':
            conv_outputs = [self.relu(conv(x)) for conv in self.convs]
            pooled = [self.global_pool(out).squeeze(-1) for out in conv_outputs]
            return torch.cat(pooled, dim=1)
        
        else: # sequential
            x = self.net(x)
            return self.global_pool(x).squeeze(-1)

# --- МОДУЛЬ 2: КВАНТОВЫЙ СЛОЙ (Re-uploading) ---
class QuantumReUploadingLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, entangling_weights, embedding_weights):
            # Первичная загрузка
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            
            for i in range(self.n_layers):
                # Обучаемое запутывание
                qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(n_qubits))
                # Re-uploading: данные * веса
                qml.AngleEmbedding(features=inputs * embedding_weights[i], wires=range(n_qubits), rotation="X")
            
            # Финальное запутывание
            qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        # Формы весов соответствуют коду из их model.py
        weight_shapes = {
            "entangling_weights": (n_layers + 1, 1, n_qubits, 3),
            "embedding_weights": (n_layers, n_qubits)
        }
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        # Масштабируем вход до [0, pi] для AngleEmbedding
        x = torch.atan(x) # или tanh * pi
        return self.qlayer(x)

# --- МОДУЛЬ 3: ГЛАВНЫЙ КЛАСС МОДЕЛИ ---
class HQDeepDTAF(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # 1. Ветки извлечения признаков (3 шт)
        self.protein_net = FlexCNNBlock(config['prot_vocab'], config['embed_dim'], mode=config['cnn_mode'])
        self.ligand_net = FlexCNNBlock(config['lig_vocab'], config['embed_dim'], mode=config['cnn_mode'])
        self.pocket_net = FlexCNNBlock(config['prot_vocab'], config['embed_dim'], mode=config['cnn_mode'])
        
        # Размер после склейки: 3 ветки * (3 ядра * 32 фильтра) = 288
        combined_dim = self.protein_net.out_dim + self.ligand_net.out_dim + self.pocket_net.out_dim
        # combined_dim = 3 * (32 * 3)
        
        # 2. Слой сжатия до размерности кубитов
        self.pre_quantum = nn.Linear(combined_dim, config['n_qubits'])
        
        # 3. Квантовое ядро
        self.quantum_core = QuantumReUploadingLayer(config['n_qubits'], config['q_layers'])
        
        # 4. Финальный регрессор
        self.regressor = nn.Sequential(
            nn.Linear(config['n_qubits'], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, prot, lig, pock):
        p_feat = self.protein_net(prot)
        l_feat = self.ligand_net(lig)
        pk_feat = self.pocket_net(pock)
        
        combined = torch.cat([p_feat, l_feat, pk_feat], dim=1)
        
        # Сжимаем и готовим для квантов
        latent = self.pre_quantum(combined)
        
        # Квантовые вычисления
        q_out = self.quantum_core(latent)
        
        # Предсказание аффинности
        return self.regressor(q_out)
