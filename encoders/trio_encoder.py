# encoders/classic_trio_encoder.py

import torch
import torch.nn as nn
from encoders.cnn_encoder import FlexCNNBlock
from encoders.gnn_encoder import FlexGNNBlock
from encoders.null_encoder import NullEncoder
from typing import Tuple, Any, Dict
from logger import logger


class TrioEncoder(nn.Module):
    """
    Trio encoder for processing protein-ligand-pocket complexes.

    Combines three separate encoders (protein, ligand, pocket) and concatenates
    their outputs. Supports different encoder types (CNN, GNN, Null) for each
    component based on configuration.

    Attributes:
        prot_enc: Encoder for protein data.
        lig_enc: Encoder for ligand data.
        pock_enc: Encoder for pocket data.
        out_dim: Total output dimension after concatenation.

    Example:
        >>> prot_enc = FlexCNNBlock(...)
        >>> lig_enc = FlexCNNBlock(...)
        >>> pock_enc = FlexGNNBlock(...)
        >>> encoder = TrioEncoder(prot_enc, lig_enc, pock_enc)
        >>> output = encoder((prot_data, lig_data, pock_data, complex_data, target))
    """

    def __init__(self, prot_enc: nn.Module, lig_enc: nn.Module, pock_enc: nn.Module) -> None:
        """
        Initialize trio encoder.

        Args:
            prot_enc: Protein encoder module.
            lig_enc: Ligand encoder module.
            pock_enc: Pocket encoder module.
        """
        super().__init__()
        self.prot_enc = prot_enc
        self.lig_enc = lig_enc
        self.pock_enc = pock_enc

        # Calculate total dimension after concatenation
        self.out_dim = self.prot_enc.out_dim + self.lig_enc.out_dim + self.pock_enc.out_dim

    def _apply_encoder(self, encoder: nn.Module, data: Any) -> torch.Tensor:
        """
        Smart router. Determines data nature (graph or string)
        and calls encoder with correct arguments.
        """
        if isinstance(encoder, NullEncoder):
            return encoder(data)
        if hasattr(data, 'edge_index'):
            # This is PyG Batch (from GNNParser)
            return encoder(x=data.x, edge_index=data.edge_index, batch=data.batch)
        else:
            # This is standard tensor (from CNNParser)
            return encoder(data)

    def forward(self, data: Tuple) -> torch.Tensor:
        """
        Forward pass through trio encoder.

        Args:
            data: Tuple of (protein, ligand, pocket, complex_graph, target).

        Returns:
            Concatenated feature vector [batch_size, out_dim].
        """
        # Expect tuple of three elements (from our Dataset)
        prot, lig, pock, _, _ = data
        
        # Micro-level: extract features (each encoder outputs flat vector [Batch, F])
        p_feat = self._apply_encoder(self.prot_enc, prot)
        l_feat = self._apply_encoder(self.lig_enc, lig)
        pk_feat = self._apply_encoder(self.pock_enc, pock)
        
        # Fusion: return macroscopic complex vector [Batch, sum(F)]
        return torch.cat([p_feat, l_feat, pk_feat], dim=1)


def build_trio_encoder(config_dict: Dict[str, Any]) -> TrioEncoder:
    """
    Factory that assembles required encoders based on string like 'CGC, GGG, NGG'.

    Parses the encoder configuration string and creates appropriate encoder
    instances (CNN, GNN, or Null) for each component.

    Args:
        config_dict: Complete configuration dictionary.

    Returns:
        Configured TrioEncoder instance.

    Example:
        >>> config = {
        ...     'model': {'graph_encoder': {'available': {'trio': {'protein_ligand_pocket_encoders': 'CGG'}}}},
        ...     'dataset': {'prot_vocab': 'ACDEFGHIKLMNPQRSTVWY', 'lig_vocab': 'ABCDEFGHIKLMNOP'}
        ... }
        >>> encoder = build_trio_encoder(config)
    """
    # 1. Navigate to trio settings
    trio_cfg = config_dict['model']['graph_encoder']['available']['trio']
    config_str = trio_cfg['protein_ligand_pocket_encoders']
    prot_v = {c: i for i, c in enumerate(config_dict['dataset']['prot_vocab'])}
    lig_v = {c: i for i, c in enumerate(config_dict['dataset']['lig_vocab'])}
    
    encoders = []
    
    # Config for each of 3 slots
    for i, char in enumerate(config_str):
        if char == 'N':
            # Null-block preserves hidden_dim so total out_dim is predictable
            enc = NullEncoder(out_dim=trio_cfg.get('hidden_dim', 128))

        elif char == 'C':
            # Get appropriate vocab depending on position (0:prot, 1:lig, 2:pock)
            vocab = lig_v if i == 1 else prot_v
            len_vocab = len(vocab)
            params = trio_cfg['cnn_params']
            enc = FlexCNNBlock(vocab_size=len_vocab, **params)
        elif char == 'G':
            params = trio_cfg['gnn_params']
            enc = FlexGNNBlock(**params)
        else:
            enc.out_dim = trio_cfg['hidden_dim']
            raise ValueError(f"Unknown encoder type: {char}")
            
        encoders.append(enc)
        
    return TrioEncoder(prot_enc=encoders[0], lig_enc=encoders[1], pock_enc=encoders[2])

# Usage:
# trio_encoder = build_trio_encoder("CGC", config)
