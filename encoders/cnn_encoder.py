# encoders/cnn_encoder.py

import torch
import torch.nn as nn
from typing import List, Dict, Any
from utils import Utils
from logger import logger


class FlexCNNBlock(nn.Module):
    """
    Flexible CNN block for sequence encoding with parallel or sequential convolution modes.

    Supports both parallel convolution filters (independent feature extraction) and
    sequential convolution layers (deep feature composition). Uses dynamic activation
    and pooling functions based on configuration.

    Attributes:
        mode: Convolution mode ('parallel' or 'sequential').
        embedding: Token embedding layer.
        act: Activation function.
        global_pool: Global pooling layer.
        convs: List of convolution layers (parallel mode).
        net: Sequential network (sequential mode).
        out_dim: Output dimension after convolution and pooling.

    Example:
        >>> cnn_block = FlexCNNBlock(vocab_size=20, embed_dim=128, filters=32, kernels=[4, 8, 12])
        >>> input_tensor = torch.randint(0, 20, (10, 50))  # batch_size=10, seq_len=50
        >>> output = cnn_block(input_tensor)
        >>> print(output.shape)  # (10, 96) for parallel mode with 3 kernels
    """

    def __init__(self, vocab_size: int, **kwargs: Any) -> None:
        """
        Initialize flexible CNN block.

        Args:
            vocab_size: Size of vocabulary for embedding layer.
            **kwargs: Configuration parameters including:
                - mode: 'parallel' or 'sequential' (default: 'parallel')
                - embed_dim: Embedding dimension (default: 128)
                - filters: Number of filters per convolution (default: 32)
                - kernels: List of kernel sizes (default: [4, 8, 12])
                - activation: Activation function name (default: 'ReLU')
                - pool_type: Pooling type name (default: 'AdaptiveMaxPool1d')
                - Additional Conv1d arguments (bias, stride, etc.)
        """
        super().__init__()
        self.mode = kwargs.get('mode', 'parallel')
        embed_dim = kwargs.get('embed_dim', 128)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        filters = kwargs.get('filters', 32)
        kernels = kwargs.get('kernels', [4, 8, 12])
        act_name = kwargs.get('activation', 'ReLU')
        pool_name = kwargs.get('pool_type', 'AdaptiveMaxPool1d')

        # Dynamic activation and pooling
        self.act = getattr(nn, act_name)()
        self.global_pool = getattr(nn, pool_name)(1)

        # Filter kwargs for Conv1d arguments
        conv_args = Utils.filter_kwargs(nn.Conv1d, kwargs)

        if self.mode == 'parallel':
            # Independent filters of different sizes
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, filters, kernel_size=k, padding=k//2, **conv_args)
                for k in kernels
            ])
            self.out_dim = filters * len(kernels)

        elif self.mode == 'sequential':
            # Deep sequential approach: layers see combinations of previous features
            layers = []
            curr_dim = embed_dim
            for k in kernels:
                layers.append(nn.Conv1d(curr_dim, filters, kernel_size=k, padding=k//2, **conv_args))
                layers.append(self.act)
                curr_dim = filters
            self.net = nn.Sequential(*layers)
            self.out_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN block.

        Args:
            x: Input tensor of shape (batch_size, seq_len) with token indices.

        Returns:
            Output tensor after convolution and pooling.

        Example:
            >>> x = torch.randint(0, 20, (5, 100))
            >>> output = cnn_block(x)
            >>> print(output.shape)  # Depends on mode and configuration
        """
        x = self.embedding(x).permute(0, 2, 1)

        if self.mode == 'parallel':
            conv_outputs = [self.act(conv(x)) for conv in self.convs]
            pooled = [self.global_pool(out).squeeze(-1) for out in conv_outputs]
            return torch.cat(pooled, dim=1)

        else:  # sequential
            x = self.net(x)
            return self.global_pool(x).squeeze(-1)
