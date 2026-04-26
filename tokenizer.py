# tokenizer.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, Any, Optional, Union, Tuple
from logger import logger


class UniversalPDBBindDataset(Dataset):
    """
    Universal dataset class for PDBBind protein-ligand complexes.

    Handles loading and preprocessing of molecular data from various formats,
    supporting both sequence-based (CNN) and graph-based (GNN) representations.
    Automatically detects file format and applies appropriate loading method.

    Attributes:
        df: DataFrame containing molecular data and affinity labels.
        prot_vocab: Vocabulary mapping for protein sequences.
        lig_vocab: Vocabulary mapping for ligand sequences.
        max_prot: Maximum protein sequence length.
        max_lig: Maximum ligand sequence length.
        max_pock: Maximum pocket atoms.

    Example:
        >>> dataset = UniversalPDBBindDataset('data/refined.parquet', config)
        >>> loader = DataLoader(dataset, batch_size=32)
        >>> for batch in loader:
        ...     # batch contains (prot, lig, pock, target)
        ...     pass
    """

    def __init__(self, df: pd.DataFrame, config_dict: Dict[str, Any]) -> None:
        """
        Initialize the dataset.

        Args:
            filepath: Path to dataset file (.csv, .pkl, .parquet).
            config_dict: Configuration dictionary with dataset parameters.

        Raises:
            ValueError: If file format is not supported.
        """
        self.df = df
        ds_cfg = config_dict['dataset']
        self.prot_vocab = {c: i for i, c in enumerate(ds_cfg['prot_vocab'])}
        self.lig_vocab = {c: i for i, c in enumerate(ds_cfg['lig_vocab'])}

        self.max_prot = ds_cfg.get('max_prot', 1000)
        self.max_lig = ds_cfg.get('max_lig', 150)
        self.max_pock = ds_cfg.get('max_pock', 63)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def _encode_sequence(self, text: Optional[str], vocab: Dict[str, int], max_len: int) -> torch.Tensor:
        """
        Encode text sequence for CNN processing.

        Args:
            text: Input text sequence or None.
            vocab: Vocabulary dictionary.
            max_len: Maximum sequence length.

        Returns:
            Encoded tensor with padding.
        """
        if text is None or pd.isna(text):
            # Return zero tensor (padding)
            return torch.zeros(max_len, dtype=torch.long)
        tokens = [vocab.get(c, 0) for c in str(text)[:max_len]]
        padded = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)

    def _encode_graph(self, graph_dict: Optional[Dict[str, Any]]) -> Data:
        """
        Encode graph dictionary for GNN processing.

        Args:
            graph_dict: Dictionary containing graph data or None.

        Returns:
            PyG Data object with encoded graph.
        """
        if graph_dict is None or not isinstance(graph_dict, dict):
            # Fallback for corrupted data
            return Data(x=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long))
            
        tensor_kwargs = {}
        for key, value in graph_dict.items():
            if key == 'edge_index':
                ei = torch.tensor(value, dtype=torch.long)
                # Исправляем размерность для PyG [2, num_edges]
                if ei.numel() > 0 and ei.shape[1] == 2 and ei.shape[0] != 2:
                    ei = ei.t().contiguous()
                tensor_kwargs[key] = ei
            elif isinstance(value, (list, np.ndarray)):
                # Если это целочисленные данные (например, атомные номера) -> long
                # Если с плавающей точкой (координаты, заряды) -> float32
                arr = np.array(value)
                dtype = torch.long if arr.dtype.kind in 'iu' else torch.float32
                tensor_kwargs[key] = torch.tensor(arr, dtype=dtype)
            else:
                tensor_kwargs[key] = value
                
        return Data(**tensor_kwargs)

    def _process_item(self, item: Any, vocab: Optional[Dict[str, int]] = None, max_len: Optional[int] = None) -> Union[torch.Tensor, Data]:
        """
        Smart router: determines data type and calls appropriate encoder.

        Args:
            item: Input data (dict for graphs, string for sequences).
            vocab: Vocabulary for sequence encoding.
            max_len: Maximum length for sequence encoding.

        Returns:
            Encoded data (tensor for sequences, Data for graphs).
        """
        if isinstance(item, dict):
            return self._encode_graph(item)
        else:
            return self._encode_sequence(item, vocab, max_len)

    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, Data], Union[torch.Tensor, Data], Union[torch.Tensor, Data], Data, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (protein_data, ligand_data, pocket_data, complex_data, target).
            Data types depend on the encoding (tensors for sequences, Data for graphs).
        """
        row = self.df.iloc[idx]
        
        # Tokenizer will figure out what's inside (string or graph)
        prot_data = self._process_item(row['protein'], self.prot_vocab, self.max_prot)
        lig_data  = self._process_item(row['ligand'], self.lig_vocab, self.max_lig)
        pock_data = self._process_item(row['pocket'], self.prot_vocab, self.max_pock)

        # Handle complex graph (for InteractionGraph/EGNN)
        complex_data = None
        if 'complex_graph' in row and row['complex_graph'] is not None:
            complex_data = self._encode_graph(row['complex_graph'])
        else:
            # If no complex graph, return empty Data
            complex_data = self._encode_graph(None)

        # Target (affinity)
        target = torch.tensor(row['pkd'], dtype=torch.float32)
        
        return prot_data, lig_data, pock_data, complex_data, target
