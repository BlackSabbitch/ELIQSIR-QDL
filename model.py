# model.py

import torch
import torch.nn as nn


class UniversalHybridSlotModel(nn.Module):
    def __init__(self,
                 graph_encoder,          
                 classic_pooler,         # Слот 2A: Пулинг для классики (опционально)
                 quantum_pooler,         # Слот 2B: Пулинг до кубитов (обязателен для квантов, если размер графа > кубитов)
                 global_readout,         
                 quantum_encoder,        
                 decider_hidden_dims=[64, 32]):

        super().__init__()
        self.graph_encoder = graph_encoder
        self.classic_pooler = classic_pooler
        self.quantum_pooler = quantum_pooler
        self.readout = global_readout
        self.quantum_encoder = quantum_encoder
        
        self.history = {'train_loss': [], 'val_rmse': [], 'val_pearson': [], 'val_ci': [], 'best_y_true': None, 'best_y_pred': None}
        graph_out_dim = self.graph_encoder.out_dim

        # === 1. КЛАССИЧЕСКИЙ РЕШАТЕЛЬ ===
        self.classic_decider = self._build_mlp(graph_out_dim, decider_hidden_dims)

        # === 2. КВАНТОВЫЙ РЕШАТЕЛЬ ===
        if self.quantum_encoder is not None:
            self.to_quantum_adapter = nn.Linear(graph_out_dim, self.quantum_encoder.n_qubits)
            
            self.quantum_decider = nn.Sequential(
                nn.Linear(self.quantum_encoder.n_qubits, 16),
                nn.Tanh(),
                nn.Linear(16, 1)
            )
            
            # === 3. ОБУЧАЕМОЕ СЛИЯНИЕ (Твоя идея!) ===
            # Вместо жесткого '+', мы учим веса: w1 * Base + w2 * Shift + Bias
            self.final_mixer = nn.Linear(2, 1)
            # Инициализируем так, чтобы на старте это было точное сложение (1.0 * base + 1.0 * shift + 0.0)
            nn.init.constant_(self.final_mixer.weight, 1.0)
            nn.init.constant_(self.final_mixer.bias, 0.0)

    def _build_mlp(self, in_features, hidden_dims):
        layers = []
        curr_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1)) 
        return nn.Sequential(*layers)

    def forward(self, data):
        node_features = self.graph_encoder(data)

        # === КЛАССИЧЕСКАЯ ВЕТКА ===
        classic_nodes = node_features
        if self.classic_pooler is not None:
            classic_nodes = self.classic_pooler(node_features)
            
        global_classic = self.readout(classic_nodes)
        base_affinity = self.classic_decider(global_classic) # Скаляр [Batch, 1]

        if self.quantum_encoder is None:
            return base_affinity

        # === КВАНТОВАЯ ВЕТКА ===
        quantum_nodes = node_features
        if self.quantum_pooler is not None:
            # Специфичный квантовый пулинг (например, поиск 9 горячих точек)
            quantum_nodes = self.quantum_pooler(node_features)
            q_context = self.to_quantum_adapter(quantum_nodes).squeeze(-1) 
        else:
            q_context = self.to_quantum_adapter(global_classic)

        q_features = self.quantum_encoder(q_context)
        quantum_affinity_shift = self.quantum_decider(q_features) # Скаляр [Batch, 1]

        # === ОБУЧАЕМАЯ ТЕОРИЯ ВОЗМУЩЕНИЙ ===
        # Конкатенируем два скаляра: [Batch, 2]
        energies = torch.cat([base_affinity, quantum_affinity_shift], dim=1)
        
        # Миксер сам найдет оптимальные веса (например, 0.95 и 1.05)
        return self.final_mixer(energies)
