# esm_enricher.py

import torch
import esm
import numpy as np
from typing import Optional
from logger import log_info


class ESMEnricher:
    """
    ESM (Evolutionary Scale Modeling) enricher for protein sequence embeddings.

    Uses pre-trained ESM models to generate contextual embeddings for protein
    sequences. Provides sequence-level representations suitable for downstream
    tasks like binding affinity prediction.

    Attributes:
        model: Loaded ESM model.
        alphabet: ESM alphabet for tokenization.
        device: Device for computation.
        batch_converter: Converter for batch processing.

    Example:
        >>> enricher = ESMEnricher(model_name="esm2_t33_650M_UR50D")
        >>> embedding = enricher.get_sequence_embedding("MKLVLSLSLLVLVL")
        >>> print(embedding.shape)  # (1280,)
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = "cuda") -> None:
        """
        Initialize ESM enricher.

        Args:
            model_name: Name of pre-trained ESM model to load.
            device: Device for computation ('cuda' or 'cpu').
        """
        log_info(f"Loading ESM model {model_name}...", stage="ESM")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def get_sequence_embedding(self, sequence: str) -> np.ndarray:
        """
        Get vector representation of entire sequence (for CNN branch).

        Args:
            sequence: Protein sequence string.

        Returns:
            Sequence embedding as numpy array.

        Example:
            >>> embedding = enricher.get_sequence_embedding("ACDEFGHIK")
            >>> print(embedding.shape)  # (1280,)
        """
        # Prepare data
        data = [("seq", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Run through ESM
        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # Average embeddings of all amino acids to get protein vector (size 1280)
        sequence_representation = token_representations[0, 1 : -1].mean(0)
        return sequence_representation.cpu().numpy()