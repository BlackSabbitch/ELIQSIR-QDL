# model/model.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from logger import *


class UniversalHybridSlotModel(nn.Module):
    """
    Universal Hybrid Slot Model for protein-ligand binding affinity prediction.

    This model implements a hybrid classical-quantum architecture that combines
    graph-based encoders for molecular structures with optional quantum perturbation
    components. The model processes protein, ligand, and pocket data through
    configurable encoder slots and produces binding affinity predictions.

    Attributes:
        graph_encoder (nn.Module): Encoder for processing input molecular graphs.
        classic_head (Optional[nn.Module]): Classical head for base affinity prediction.
        quantum_encoder (Optional[nn.Module]): Quantum encoder for perturbation calculations.
        to_quantum_adapter (Optional[nn.Module]): Adapter to transform classical features to quantum inputs.
        quantum_head (Optional[nn.Module]): Quantum head for processing quantum outputs.
        final_mixer (Optional[nn.Module]): Final mixer combining classical and quantum predictions.

    Example:
        >>> config = load_config('config.json')
        >>> model = UHSMBuilder.build_model_from_config(config)
        >>> # Assuming data is a tuple (protein, ligand, pocket)
        >>> prediction = model(data)
        >>> print(prediction.shape)  # torch.Size([batch_size, 1])
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        # --- Classical branch (Base Affinity) ---
        classic_head: Optional[nn.Module] = None,
        # --- Quantum branch (Perturbation Shift) ---
        to_quantum_adapter: Optional[nn.Module] = None,
        quantum_encoder: Optional[nn.Module] = None,
        quantum_head: Optional[nn.Module] = None,
        # --- Fusion ---
        final_mixer: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        self.graph_encoder = graph_encoder
        
        # Классика
        # self.classic_node_processor = classic_node_processor
        # self.classic_aggregation = classic_aggregation
        self.classic_head = classic_head

        # Кванты
        # self.quantum_node_processor = quantum_node_processor
        # self.quantum_aggregation = quantum_aggregation
        self.quantum_encoder = quantum_encoder
        self.to_quantum_adapter = to_quantum_adapter
        self.quantum_head = quantum_head
        self.final_mixer = final_mixer

        # Auto-initialization and checks
        if self.quantum_encoder is not None:
            self._init_submodules_for_quantum_branch()

    def _init_submodules_for_quantum_branch(self) -> None:
        """
        Initialize default submodules for the quantum branch.

        Automatically creates necessary adapters, heads, and mixers if not provided.
        Sets up default initialization for quantum components.

        This method ensures that all quantum-related components are properly
        configured with appropriate dimensions and initial weights.
        """
        # If adapter not provided, create linear layer based on encoder output dimension
        if self.to_quantum_adapter is None:
            in_dim = getattr(self.graph_encoder, 'out_dim', 128)  # fallback dimension
            self.to_quantum_adapter = nn.Linear(in_dim, self.quantum_encoder.in_dim)

        # If quantum head not provided - standard Tanh-regressor
        if self.quantum_head is None:
            self.quantum_head = nn.Sequential(
                nn.Tanh(), 
                nn.Linear(self.quantum_encoder.in_dim, 1)
            )

        # Setup mixer (Perturbation Theory)
        if self.final_mixer is None:
            self.final_mixer = nn.Linear(2, 1)
            with torch.no_grad():
                # At start: 1.0 * base + 1.0 * shift + 0.0
                nn.init.constant_(self.final_mixer.weight, 1.0)
                nn.init.constant_(self.final_mixer.bias, 0.0)

    def _safe_call(self, module: Optional[nn.Module], x: torch.Tensor, data) -> torch.Tensor:
        """
        Safely call pooling/aggregation modules with different signatures.

        Attempts to call the module with different argument patterns to handle
        various PyTorch Geometric pooling operations.

        Args:
            module: The module to call (pooling or aggregation layer).
            x: Node features tensor.
            data: PyG Data object containing edge_index and batch information.

        Returns:
            Processed tensor output from the module.

        Raises:
            Logs warnings if module call fails but continues with fallback.
        """
        if module is None:
            return x
        try:
            # Try calling as complex pooling (x, edge_index, batch)
            out = module(x, data.edge_index, data.batch)
        except (TypeError, AttributeError):
            # Try calling as simple readout (x, batch)
            out = module(x, data.batch)
        
        # Handle tuples from PyG layers (TopK, SAGPool)
        return out[0] if isinstance(out, tuple) else out

    def _call_module_with_optional_progress(self, module: Optional[nn.Module], x: torch.Tensor, progress: float) -> torch.Tensor:
        """
        Call module with progress if supported, otherwise call normally.

        This allows classic heads like VQE wrappers to receive schedule progress
        without breaking standard MLP/Linear modules.
        """
        if module is None:
            return x
        try:
            return module(x, progress=progress)
        except TypeError:
            return module(x)

    def forward(self, data: Tuple, progress: float = 0.0) -> torch.Tensor:
        """
        Forward pass through the hybrid model.

        Processes input molecular data through classical and optional quantum branches
        to produce binding affinity predictions.

        Args:
            data: Tuple containing (protein_data, ligand_data, pocket_data).
                  Each element should be compatible with the graph_encoder input format.

        Returns:
            torch.Tensor: Predicted binding affinity values with shape [batch_size, 1].
                         If quantum branch is disabled, returns classical prediction.
                         If enabled, returns combined classical + quantum prediction.

        Example:
            >>> # Assuming data is prepared as (prot_batch, lig_batch, pock_batch)
            >>> output = model(data)
            >>> print(output.shape)  # torch.Size([32, 1])
        """
        # 0. General encoding: [Nodes, Hidden]
        features = self.graph_encoder(data)

        # === CLASSICAL BRANCH ===
        # 3A. Solver -> Base affinity
        base_affinity = self._call_module_with_optional_progress(self.classic_head, features, progress)  # [Batch, 1]

        if self.quantum_encoder is None:
            return base_affinity

        # === QUANTUM BRANCH ===
        z_quantum = features
        # 3B. Adapter (angles for qubits)
        q_context = self.to_quantum_adapter(z_quantum)
        # 4B. Quantum core (Data Re-uploading)
        q_features = self.quantum_encoder(q_context, progress=progress)
        # 5B. Quantum correction
        q_affinity_shift = self.quantum_head(q_features)  # [Batch, 1]

        # === MIXER ===
        # Concatenate along last dimension (features)
        combined = torch.cat([base_affinity, q_affinity_shift], dim=-1)
        return self.final_mixer(combined)
