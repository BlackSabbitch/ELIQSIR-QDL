# splitter.py

import numpy as np
import random
import hashlib
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


class PDBBindSplitter:
    """
    Набор стратегий разбиения PDBBind refined dataset.
    Все методы возвращают (train_df, val_df)
    """

    # ======================
    # RANDOM SPLIT
    # ======================
    @staticmethod
    def random_split(df, val_frac=0.1, seed=42):
        np.random.seed(seed)
        indices = np.random.permutation(len(df))

        val_size = int(len(df) * val_frac)
        val_idx = indices[:val_size]
        train_idx = indices[val_size:]

        return df.iloc[train_idx], df.iloc[val_idx]

    # ======================
    # SCAFFOLD SPLIT
    # ======================
    @staticmethod
    def _get_scaffold(smiles):
        """
        Возвращает Murcko scaffold или None, если не удалось построить.
        """
        if not isinstance(smiles, str) or len(smiles) == 0:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None:
                return None

            # Канонизация без стереохимии (стандартная практика)
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)

        except Exception:
            return None

    @staticmethod
    def scaffold_split_strict(df, val_frac=0.1, seed=42):
        random.seed(seed)

        scaffold_to_indices = defaultdict(list)

        for i, smi in enumerate(df['smiles']):
            scaf = PDBBindSplitter._get_scaffold(smi)

            if scaf is None:
                scaf = f"NO_SCAF_{i}"

            scaffold_to_indices[scaf].append(i)

        # сортировка по размеру (большие сначала)
        scaffolds = sorted(scaffold_to_indices.values(), key=len, reverse=True)

        train_idx, val_idx = [], []
        val_target = int(len(df) * val_frac)

        for group in scaffolds:
            if len(val_idx) + len(group) <= val_target:
                val_idx.extend(group)
            else:
                train_idx.extend(group)

        return df.iloc[train_idx], df.iloc[val_idx]

    @staticmethod
    def scaffold_split_balanced(df, val_frac=0.1, seed=42):
        random.seed(seed)

        scaffold_to_indices = defaultdict(list)

        for i, smi in enumerate(df['smiles']):
            scaf = PDBBindSplitter._get_scaffold(smi)

            if scaf is None:
                scaf = f"NO_SCAF_{i}"

            scaffold_to_indices[scaf].append(i)

        scaffolds = sorted(scaffold_to_indices.values(), key=len, reverse=True)

        train_idx, val_idx = [], []
        train_target = int(len(df) * (1 - val_frac))

        for group in scaffolds:
            if len(train_idx) + len(group) <= train_target:
                train_idx.extend(group)
            else:
                val_idx.extend(group)

        return df.iloc[train_idx], df.iloc[val_idx]

    @staticmethod
    def cold_protein_split(df, val_frac=0.1, seed=42):
        random.seed(seed)

        protein_to_indices = defaultdict(list)

        for i, seq in enumerate(df['seq']):
            protein_to_indices[seq].append(i)

        groups = list(protein_to_indices.values())
        random.shuffle(groups)

        train_idx, val_idx = [], []
        val_target = int(len(df) * val_frac)

        for group in groups:
            if len(val_idx) + len(group) <= val_target:
                val_idx.extend(group)
            else:
                train_idx.extend(group)

        return df.iloc[train_idx], df.iloc[val_idx]

    # ======================
    # UNIFIED INTERFACE
    # ======================
    @staticmethod
    def split(df, strategy="random", val_frac=0.1, seed=42):
        print(f"Split strategy: {strategy}")

        if strategy == "random":
            return PDBBindSplitter.random_split(df, val_frac, seed)

        elif strategy == "scaffold":
            return PDBBindSplitter.scaffold_split_strict(df, val_frac, seed)

        elif strategy == "scaffold_balanced":
            return PDBBindSplitter.scaffold_split_balanced(df, val_frac, seed)

        elif strategy == "cold_protein":
            return PDBBindSplitter.cold_protein_split(df, val_frac, seed)

        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
