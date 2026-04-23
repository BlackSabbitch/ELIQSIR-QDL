# encoders/null_encoder.py

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any
from logger import logger


class NullEncoder(nn.Module):
    """
    Placeholder encoder that outputs zero embeddings for missing branches.

    Useful when one input modality is disabled or absent, but model structure
    requires a fixed-size embedding vector.

    Example:
        >>> encoder = NullEncoder(out_dim=128)
        >>> data = torch.zeros((8, 1))
        >>> output = encoder(data)
        >>> print(output.shape)  # torch.Size([8, 128])
    """

    def __init__(self, out_dim: int = 128) -> None:
        """
        Initialize the null encoder.

        Args:
            out_dim: Output embedding dimension.
        """
        super().__init__()
        self.out_dim = out_dim
        logger.info(f"[NullEncoder] Initialized with out_dim={out_dim}")

    def forward(self, data: Any) -> Tensor:
        """
        Forward pass returning zero embeddings.

        Args:
            data: Input data used only to infer batch size and device.

        Returns:
            Zero tensor of shape [batch_size, out_dim].
        """
        if hasattr(data, 'shape'):
            batch_size = data.shape[0]
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            batch_size = data[0].shape[0]
        else:
            raise ValueError("Unable to infer batch size from NullEncoder input")

        device = getattr(data, 'device', torch.device('cpu'))
        return torch.zeros((batch_size, self.out_dim), device=device)
