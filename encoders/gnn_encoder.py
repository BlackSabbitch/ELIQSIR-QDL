# encoders/gnn_encoder.py

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_max_pool, global_mean_pool, GATConv, GATv2Conv, TransformerConv, SAGPooling, TopKPooling
from typing import Any, Dict, Optional
from utils import Utils
from logger import logger


class FlexGNNBlock(nn.Module):
    """
    Flexible GNN block for node feature encoding and graph-level pooling.

    Supports multiple graph convolution types and optional hierarchical pooling.
    Designed to handle both protein and ligand graphs with customizable input adapters.

    Example:
        >>> block = FlexGNNBlock(in_channels=4, out_channels=128, num_layers=3, conv_type='gat_v2')
        >>> x = torch.randn((32, 4))
        >>> edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        >>> batch = torch.zeros(32, dtype=torch.long)
        >>> out = block(x, edge_index, batch)
        >>> print(out.shape)
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize flexible GNN block.

        Args:
            **kwargs: Configuration for GNN block, including:
                - in_channels: Input node feature dimension.
                - hidden_channels: Hidden dimension for intermediate conv layers.
                - out_channels: Output node embedding dimension.
                - num_layers: Number of graph convolution layers.
                - conv_type: Graph convolution type ('gat', 'gat_v2', 'transformer', 'egnn').
                - activation: Activation name for intermediate layers.
                - node_processor: Hierarchical pooling settings.
                - aggregation: Graph pooling type settings.
                - use_batch_norm: Whether to apply BatchNorm1d.
        """
        super().__init__()
        heads = kwargs.get('heads', 4)
        in_channels = kwargs.get('in_channels', 4)
        out_channels = kwargs.get('out_channels', 128)

        if in_channels == 1:
            self.input_adapter = nn.Embedding(100, out_channels)
        else:
            self.input_adapter = nn.Linear(in_channels, out_channels)

        num_layers = kwargs.get('num_layers', 3)
        self.conv_type = kwargs.get('conv_type', 'gat_v2').lower()
        hidden_channels = kwargs.get('hidden_channels', out_channels)
        act_name = kwargs.get('activation', 'ReLU')
        self.act = getattr(nn.functional, act_name.lower())

        node_processor_settings = kwargs.get('node_processor', {})
        hier_pool_type = node_processor_settings.get('selected', 'none')
        hier_pool_args = node_processor_settings.get('available', {}).get(hier_pool_type, {})
        aggregation_settings = kwargs.get('aggregation', {})
        self.pool_type = aggregation_settings.get('selected', 'max')
        self.pool_args = aggregation_settings.get('available', {}).get(self.pool_type, {})

        self.use_bn = kwargs.get('use_batch_norm', True)

        self.convs = nn.ModuleList()

        conv_map = {
            'gat': GATConv,
            'gat_v2': GATv2Conv,
            'transformer': TransformerConv,
            'egnn': None,
        }
        conv_class = conv_map.get(self.conv_type, GATv2Conv)
        actual_conv_args = Utils.filter_kwargs(conv_class or GATv2Conv, kwargs)
        # Prevent duplicate explicit arguments when building convolutions.
        for forbidden_key in ['in_channels', 'out_channels', 'heads', 'concat']:
            actual_conv_args.pop(forbidden_key, None)

        if self.conv_type != 'egnn':
            curr_dim = in_channels
            for i in range(num_layers):
                is_last = i == num_layers - 1
                if is_last:
                    self.convs.append(conv_class(curr_dim, out_channels, heads=heads, concat=False, **actual_conv_args))
                else:
                    h_dim = hidden_channels // heads
                    self.convs.append(conv_class(curr_dim, h_dim, heads=heads, concat=True, **actual_conv_args))
                    curr_dim = h_dim * heads
        else:
            logger.info("[GNN] EGNN special case selected; no standard GNN conv layers are built.")

        if hier_pool_type == 'sag':
            self.hier_pool = SAGPooling(out_channels, **hier_pool_args)
        elif hier_pool_type == 'topk':
            self.hier_pool = TopKPooling(out_channels, **hier_pool_args)
        else:
            self.hier_pool = None

        self.batch_norm = nn.BatchNorm1d(out_channels) if self.use_bn else nn.Identity()
        self.out_dim = out_channels

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """
        Forward pass through the graph encoder.

        Args:
            x: Node features tensor with shape [num_nodes, in_channels].
            edge_index: Edge indices tensor with shape [2, num_edges].
            batch: Batch assignment tensor for pooling.

        Returns:
            Graph-level embedding tensor after pooling.
        """
        x = x.float()
        if self.conv_type != 'egnn':
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i != len(self.convs) - 1:
                    x = self.act(x)

        x = self.batch_norm(x)

        if self.pool_type == 'max':
            return global_max_pool(x, batch)
        if self.pool_type == 'mean':
            return global_mean_pool(x, batch)
        if self.pool_type is None:
            return x
        return x
