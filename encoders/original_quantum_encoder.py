# encoders/original_quantum_encoder.py

import torch
import torch.nn as nn
import pennylane as qml


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
        # Масштабируем вход до [-pi, pi] для AngleEmbedding
        x = torch.tanh(x) * torch.pi
        return self.qlayer(x)
