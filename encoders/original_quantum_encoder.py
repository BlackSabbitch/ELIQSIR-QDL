# encoders/original_quantum_encoder.py

import torch
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from typing import Any, Dict
from logger import log_info, log_warn
from utils import Utils


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

    def __init__(self, in_dim: int, n_layers: int, **kwargs: Any) -> None:
        """
        Initialize the quantum re-uploading layer.

        Args:
            in_dim: Number of quantum wires/qubits.
            n_layers: Number of variational layers in the circuit.
            **kwargs: Additional PennyLane-specific configuration options.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_layers = n_layers

        self.rotation = kwargs.pop("rotation", "X")
        entanglement = kwargs.pop("entanglement", "full")
        self.entanglement = "full" if entanglement == "strongly_entangling" else entanglement

        self.angle_embedding_kwargs: Dict[str, Any] = Utils.filter_kwargs(qml.AngleEmbedding, kwargs)
        self.strongly_entangling_kwargs: Dict[str, Any] = Utils.filter_kwargs(qml.StronglyEntanglingLayers, kwargs)

        ignored_kwargs = {k: v for k, v in kwargs.items() if k not in self.angle_embedding_kwargs and k not in self.strongly_entangling_kwargs}
        if ignored_kwargs:
            log_warn(
                f"Ignored unsupported QuantumReUploadingLayer kwargs: {list(ignored_kwargs.keys())}",
                stage="QuantumReUploadingLayer"
            )

        log_info(
            f"Creating with {in_dim} qubits, {n_layers} layers, rotation={self.rotation}, entanglement={self.entanglement}",
            stage="QuantumReUploadingLayer"
        )
        dev = qml.device("default.qubit", wires=in_dim)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: Tensor, entangling_weights: Tensor, embedding_weights: Tensor) -> Any:
            qml.AngleEmbedding(
                inputs,
                wires=range(in_dim),
                rotation=self.rotation,
                **self.angle_embedding_kwargs
            )
            for i in range(self.n_layers):
                qml.StronglyEntanglingLayers(
                    entangling_weights[i],
                    wires=range(in_dim),
                    rotation=self.rotation,
                    entanglement=self.entanglement,
                    **self.strongly_entangling_kwargs
                )
                qml.AngleEmbedding(
                    features=inputs * embedding_weights[i],
                    wires=range(in_dim),
                    rotation=self.rotation,
                    **self.angle_embedding_kwargs
                )
            qml.StronglyEntanglingLayers(
                entangling_weights[-1],
                wires=range(in_dim),
                rotation=self.rotation,
                entanglement=self.entanglement,
                **self.strongly_entangling_kwargs
            )
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
