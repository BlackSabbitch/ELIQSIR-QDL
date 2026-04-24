# encoders/original_quantum_encoder.py

import torch
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from typing import Any
from logger import log_info


class QuantumReUploadingLayer(nn.Module):
    """
    Quantum layer implementing re-uploading of classical inputs into a parametric quantum circuit.

    This module wraps a PennyLane TorchLayer and scales inputs into the required range for
    AngleEmbedding. It is intended for integration with hybrid classical-quantum models.

    Example:
        >>> qlayer = QuantumReUploadingLayer(in_dim=4, n_layers=2)
        >>> x = torch.randn((1, 4))
        >>> y = qlayer(x)
        >>> print(y.shape)  # torch.Size([1, 4])
    """

    def __init__(self, in_dim: int, n_layers: int) -> None:
        """
        Initialize the quantum re-uploading layer.

        Args:
            in_dim: Number of quantum wires/qubits.
            n_layers: Number of variational layers in the circuit.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_layers = n_layers

        log_info(f"Creating with {in_dim} qubits and {n_layers} layers", stage="QuantumReUploadingLayer")
        dev = qml.device("default.qubit", wires=in_dim)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: Tensor, entangling_weights: Tensor, embedding_weights: Tensor) -> Any:
            qml.AngleEmbedding(inputs, wires=range(in_dim), rotation="X")
            for i in range(self.n_layers):
                qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(in_dim))
                qml.AngleEmbedding(features=inputs * embedding_weights[i], wires=range(in_dim), rotation="X")
            qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(in_dim))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(in_dim)]

        weight_shapes = {
            "entangling_weights": (n_layers + 1, 1, in_dim, 3),
            "embedding_weights": (n_layers, in_dim),
        }
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            x: Input tensor of shape [batch_size, in_dim].

        Returns:
            Tensor of expectation values with shape [batch_size, in_dim].
        """
        x = torch.tanh(x) * torch.pi
        return self.qlayer(x)
