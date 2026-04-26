# encoders/original_quantum_encoder.py

import torch
import torch.nn as nn
import pennylane as qml
from torch import Tensor
from typing import Any, Dict, Optional
from logger import *
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
                Supported extra kwargs:
                    input_scale: float multiplier applied before tanh.
                    start_scale: initial angle scale in radians.
                    end_scale: final angle scale in radians.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_layers = n_layers

        self.input_scale = kwargs.pop("input_scale", 0.01)
        self.scale_start = kwargs.pop("start_scale", torch.pi / 6)
        self.scale_end = kwargs.pop("end_scale", torch.pi)

        self.rotation = kwargs.pop("rotation", "X")
        self.initial_rotation = kwargs.pop("initial_rotation", "Y")
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
            f"Creating with {in_dim} qubits, {n_layers} layers, initial_rotation={self.initial_rotation}, rotation={self.rotation}, entanglement={self.entanglement}",
            stage="QuantumReUploadingLayer"
        )
        dev = qml.device("default.qubit", wires=in_dim)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs: Tensor, entangling_weights: Tensor, embedding_weights: Tensor) -> Any:
            qml.AngleEmbedding(
                inputs,
                wires=range(in_dim),
                rotation=self.initial_rotation,
                **self.angle_embedding_kwargs
            )
            for i in range(self.n_layers):
                qml.StronglyEntanglingLayers(
                    entangling_weights[i],
                    wires=range(in_dim),
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
                **self.strongly_entangling_kwargs
            )
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(in_dim)]

        weight_shapes = {
            "entangling_weights": (n_layers + 1, 1, in_dim, 3),
            "embedding_weights": (n_layers, in_dim),
        }
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x: Tensor, progress: float = 0.0) -> Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            x: Input tensor of shape [batch_size, in_dim].
            progress: Schedule progress in [0, 1] used to scale the embedded angles.

        Returns:
            Tensor of expectation values with shape [batch_size, in_dim].
        """
        progress = float(progress)
        progress = min(max(progress, 0.0), 1.0)
        scale = self.scale_start + (self.scale_end - self.scale_start) * progress
        if torch.rand(1) < 0.02:
            log_debug(f"mean: {x.mean().item():.3f}, std: {x.std().item():.3f}, max: {x.max().item():.3f}, scale: {scale:.3f}, progress: {progress:.3f}", stage="QUANTUM INPUTS")
        # x = x * self.input_scale
        x = torch.tanh(x) * scale
        return self.qlayer(x)


class VQEHead(nn.Module):
    def __init__(self, adapter: nn.Module, q_layer: nn.Module, final_layer: nn.Module) -> None:
        super().__init__()
        self.adapter = adapter
        self.q_layer = q_layer
        self.final_layer = final_layer

    def forward(self, x: torch.Tensor, progress: float = 0.0) -> torch.Tensor:
        x = self.adapter(x)
        x = self.q_layer(x, progress=progress)
        return self.final_layer(x)
