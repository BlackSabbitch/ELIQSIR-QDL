# parsers/interaction_graph_parser.py

import io
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from rdkit import Chem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from ._base_parser import BaseParser
from logger import logger


class InteractionGraphParser(BaseParser):
    """
    Parser that builds a single interaction graph combining ligand and pocket atoms.

    Produces a graph dictionary with node features, 3D positions, and edge indices.
    """

    def __init__(self, dist_threshold: float = 5.0, ca_only: bool = False) -> None:
        self.dist_threshold = dist_threshold
        self.ca_only = ca_only
        self.pdb_parser = PDBParser(QUIET=True)
        logger.info(f"[InteractionGraphParser] Initialized dist_threshold={dist_threshold}, ca_only={ca_only}")

    def parse_file(self, lig_path: str, pock_path: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse ligand and pocket files from disk.
        """
        if pock_path is None:
            return None, "InteractionGraphParser requires lig_path and pock_path"
        try:
            return self._build_complex_graph(lig_path, pock_path, is_file=True)
        except Exception as e:
            logger.warning(f"[InteractionGraphParser] parse_file failure: {e}")
            return None, str(e)

    def parse_stream(self, lig_bytes: bytes, pock_bytes: Optional[bytes] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse ligand and pocket structures from byte streams.
        """
        if pock_bytes is None:
            return None, "InteractionGraphParser requires lig_bytes and pock_bytes"
        try:
            return self._build_complex_graph(lig_bytes, pock_bytes, is_file=False)
        except Exception as e:
            logger.warning(f"[InteractionGraphParser] parse_stream failure: {e}")
            return None, str(e)

    def _build_complex_graph(self, lig_data: Any, pock_data: Any, is_file: bool = True) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if is_file:
            lig_mol = Chem.MolFromMolFile(lig_data, sanitize=False)
        else:
            lig_content = lig_data.decode('utf-8') if isinstance(lig_data, bytes) else lig_data
            lig_mol = Chem.MolFromMolBlock(lig_content, sanitize=False)

        if not lig_mol:
            return None, "ligand_load_error"

        try:
            Chem.SanitizeMol(
                lig_mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        except Exception:
            pass

        lig_coords = lig_mol.GetConformer().GetPositions()
        lig_x = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic()), 1] for a in lig_mol.GetAtoms()]

        pock_coords: List[List[float]] = []
        pock_x: List[List[int]] = []

        if is_file:
            struct = self.pdb_parser.get_structure("pock", pock_data)
        else:
            stream = io.StringIO(pock_data.decode('utf-8') if isinstance(pock_data, bytes) else pock_data)
            struct = self.pdb_parser.get_structure("pock", stream)

        for model in struct:
            for chain in model:
                for residue in chain:
                    atoms_to_process = [residue['CA']] if self.ca_only and 'CA' in residue else residue.get_atoms()
                    for atom in atoms_to_process:
                        if not self.ca_only and atom.element == 'H':
                            continue
                        coord = atom.get_coord()
                        pock_coords.append([float(coord[0]), float(coord[1]), float(coord[2])])
                        atomic_num = 6 if atom.element == 'C' else 7 if atom.element == 'N' else 8 if atom.element == 'O' else 16 if atom.element == 'S' else 0
                        pock_x.append([atomic_num, 0, 0, 0])
            break

        if not pock_coords:
            return None, "empty_pocket_ca" if self.ca_only else "empty_pocket"

        all_x = np.array(lig_x + pock_x)
        all_coords = np.concatenate([lig_coords, np.array(pock_coords)], axis=0)

        edges: List[List[int]] = []
        for bond in lig_mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([[i, j], [j, i]])

        dist_mat = cdist(lig_coords, pock_coords)
        lig_idx, pock_idx = np.where(dist_mat < self.dist_threshold)
        for i, j in zip(lig_idx, pock_idx):
            p_idx_shifted = j + len(lig_x)
            edges.extend([[i, p_idx_shifted], [p_idx_shifted, i]])

        return {
            'x': all_x.tolist(),
            'pos': all_coords.tolist(),
            'edge_index': edges,
        }, None
