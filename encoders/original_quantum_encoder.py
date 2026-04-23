# encoders/original_quantum_encoder.py

import torch
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from typing import Any
from logger import logger


class QuantumReUploadingLayer(nn.Module):
    """
    Quantum layer implementing re-uploading of classical inputs into a parametric quantum circuit.

    This module wraps a PennyLane TorchLayer and scales inputs into the required range for
    AngleEmbedding. It is intended for integration with hybrid classical-quantum models.

    Example:
        >>> qlayer = QuantumReUploadingLayer(n_qubits=4, n_layers=2)
        >>> x = torch.randn((1, 4))
        >>> y = qlayer(x)
        >>> print(y.shape)  # torch.Size([1, 4])
    """

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        """
        Initialize the quantum re-uploading layer.

        Args:
            n_qubits: Number of quantum wires/qubits.
            n_layers: Number of variational layers in the circuit.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        logger.info(f"[QuantumReUploadingLayer] Creating with {n_qubits} qubits and {n_layers} layers")
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: Tensor, entangling_weights: Tensor, embedding_weights: Tensor) -> Any:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            for i in range(self.n_layers):
                qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(n_qubits))
                qml.AngleEmbedding(features=inputs * embedding_weights[i], wires=range(n_qubits), rotation="X")
            qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        weight_shapes = {
            "entangling_weights": (n_layers + 1, 1, n_qubits, 3),
            "embedding_weights": (n_layers, n_qubits),
        }
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            x: Input tensor of shape [batch_size, n_qubits].

        Returns:
            Tensor of expectation values with shape [batch_size, n_qubits].
        """
        x = torch.tanh(x) * torch.pi
        return self.qlayer(x)
