# model.py

import torch
import torch.nn as nn
# from torch_geometric.data import Batch


class UniversalHybridModel(nn.Module):
    def __init__(self, protein_encoder, ligand_encoder, pocket_encoder, quantum_encoder):
        super().__init__()
        self.protein_encoder = protein_encoder
        self.ligand_encoder = ligand_encoder
        self.pocket_encoder = pocket_encoder
        self.quantum_encoder = quantum_encoder
        self.history = {
            'train_loss': [],
            'val_rmse': [],
            'val_pearson': [],
            'val_ci': [],
            'best_y_true': None,
            'best_y_pred': None
            }

        combined_dim = self.protein_encoder.out_dim + self.ligand_encoder.out_dim + self.pocket_encoder.out_dim
        # 2. Слой сжатия до размерности кубитов
        self.pre_quantum = nn.Linear(combined_dim, self.quantum_encoder.n_qubits)
        
        # 4. Финальный регрессор
        self.regressor = nn.Sequential(
            nn.Linear(self.quantum_encoder.n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def _apply_encoder(self, encoder, data):
        """
        Умный роутер для энкодера.
        Проверяет тип входящих данных и вызывает энкодер правильным образом.
        """
        # Если данные пришли как батч графов из PyG
        if hasattr(data, 'edge_index'):
            return encoder(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif isinstance(data, dict) and 'edge_index' in data:
            return encoder(x=data['x'], edge_index=data['edge_index'], batch=data['batch'])
        # Иначе (например, тензор для CNN)
        else:
            return encoder(data)

    def forward(self, prot, lig, pock):
        # Используем наш роутер для каждого компонента
        p_feat = self._apply_encoder(self.protein_encoder, prot)
        l_feat = self._apply_encoder(self.ligand_encoder, lig)
        pk_feat = self._apply_encoder(self.pocket_encoder, pock)

        combined = torch.cat([p_feat, l_feat, pk_feat], dim=1)
        
        # Сжимаем и готовим для квантов
        latent = self.pre_quantum(combined)
        
        # Квантовые вычисления
        q_out = self.quantum_encoder(latent)
        
        # Предсказание аффинности
        return self.regressor(q_out)


"""
HQDeepDTAF = UniversalHybridModel(
    protein_encoder = FlexCNNBlock(config['prot_vocab'], config['embed_dim'], mode=config['cnn_mode']),
    ligand_encoder = FlexCNNBlock(config['lig_vocab'], config['embed_dim'], mode=config['cnn_mode']),
    pocket_encoder = FlexCNNBlock(config['prot_vocab'], config['embed_dim'], mode=config['cnn_mode']),
    quantum_core = QuantumReUploadingLayer(config['n_qubits'], config['q_layers']),
    n_qubits = config['n_qubits']
)
"""