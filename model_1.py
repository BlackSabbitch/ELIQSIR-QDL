# model_1.py

import torch
import torch.nn as nn


class UniversalHybridSlotModel(nn.Module):
    def __init__(
        self,
        graph_encoder: nn.Module,
        # Слоты классической ветки
        classic_pooler: nn.Module = None,   
        classic_encoder: nn.Module = None,  
        classic_readout: nn.Module = None,  
        # Слоты квантовой ветки
        quantum_pooler: nn.Module = None,
        to_quantum_adapter: nn.Module = None,
        quantum_encoder: nn.Module = None,
        quantum_decider: nn.Module = None,
        final_mixer: nn.Module = None
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        
        # Классическая логика
        self.classic_pooler = classic_pooler    
        self.classic_encoder = classic_encoder  
        self.classic_readout = classic_readout  

        # Квантовая логика
        self.quantum_pooler = quantum_pooler
        self.quantum_encoder = quantum_encoder

        if self.quantum_encoder is not None:
            if to_quantum_adapter is None:
                in_dim = getattr(self.classic_encoder, 'out_dim', self.graph_encoder.out_dim)
                self.to_quantum_adapter = nn.Linear(in_dim, self.quantum_encoder.n_qubits)
            else:
                self.to_quantum_adapter = to_quantum_adapter

            if quantum_decider is None:
                self.quantum_decider = nn.Sequential(
                    nn.Tanh(), 
                    nn.Linear(self.quantum_encoder.n_qubits, 1)
                )
            else:
                self.quantum_decider = quantum_decider

            if final_mixer is None:
                self.final_mixer = nn.Linear(2, 1)
                with torch.no_grad():
                    nn.init.constant_(self.final_mixer.weight, 1.0)
                    nn.init.constant_(self.final_mixer.bias, 0.0)
            else:
                self.final_mixer = final_mixer

    def forward(self, data):
        node_features = self.graph_encoder(data)

        # === КЛАССИЧЕСКАЯ ВЕТКА ===
        c_nodes = node_features
        if self.classic_pooler is not None:
            # А ЕСЛИ ТУТ ОБЫЧНЫЙ global_max/mean - не упадет?
            c_nodes = self.classic_pooler(c_nodes, data.edge_index, data.batch)
            if isinstance(c_nodes, tuple):
                c_nodes = c_nodes[0]
        
        z_classic = self.classic_encoder(c_nodes, data.batch)
        base_affinity = self.classic_readout(z_classic) # [Batch, 1]

        if self.quantum_encoder is None:
            return base_affinity

        # === КВАНТОВАЯ ВЕТКА ===
        if self.quantum_pooler is not None:
            q_nodes = self.quantum_pooler(node_features, data.edge_index, data.batch)
            if isinstance(q_nodes, tuple):
                q_nodes = q_nodes[0]
            q_context = self.to_quantum_adapter(q_nodes)
        else:
            q_context = self.to_quantum_adapter(z_classic)        
        q_features = self.quantum_encoder(q_context)
        q_affinity_shift = self.quantum_decider(q_features) # [Batch, 1]

        # === MIXER (The Perturbation Theory) ===
        combined = torch.cat([base_affinity, q_affinity_shift], dim=-1)
        
        return self.final_mixer(combined)
