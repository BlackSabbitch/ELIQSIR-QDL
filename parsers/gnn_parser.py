# parsers/gnn_parser.py

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from rdkit import Chem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from ._base_parser import BaseParser
from logger import log_info, log_warn


class GNNParser(BaseParser):
    """
    Parser for protein and ligand graph representations suitable for GNN input.
    """

    def __init__(self, is_ligand: bool = False, dist_threshold: float = 10.0, ca_only: bool = True) -> None:
        super().__init__()
        self.is_ligand = is_ligand
        self.dist_threshold = dist_threshold
        self.ca_only = ca_only
        log_info(f"Initialized is_ligand={is_ligand}, dist_threshold={dist_threshold}, ca_only={ca_only}", stage="GNNParser")

    def parse_file(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Parse a PDB or MOL file into graph data for GNN embeddings.

        Args:
            path: File path to the protein or ligand source.

        Returns:
            A tuple containing graph data and an optional error message.
        """
        try:
            if self.is_ligand:
                mol = Chem.MolFromMolFile(path, sanitize=False)
                if not mol:
                    return None, "ligand_load_error"
                return self._process_ligand(mol)
            return self._process_protein(path)
        except Exception as e:
            log_warn(f"parse_file failed: {e}", stage="GNNParser")
            return None, str(e)

    def _process_protein(self, path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        coords: List[List[float]] = []
        atomic_nums: List[List[int]] = []

        try:
            mol = Chem.MolFromPDBFile(path, sanitize=False, proximityBonding=False)
            if mol:
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    if self.ca_only:
                        info = atom.GetPDBResidueInfo()
                        if not info or info.GetName().strip() != "CA":
                            continue
                    else:
                        if atom.GetSymbol() == 'H':
                            continue
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    coords.append([pos.x, pos.y, pos.z])
                    atomic_nums.append([atom.GetAtomicNum()])
        except Exception as e:
            log_info(f"RDKit protein parse failed, falling back to Biopython: {e}", stage="GNNParser")
            coords, atomic_nums = [], []

        if not coords:
            try:
                pdb_parser = PDBParser(QUIET=True)
                struct = pdb_parser.get_structure("prot", path)
                for model in struct:
                    for chain in model:
                        for residue in chain:
                            atoms = [residue['CA']] if self.ca_only and 'CA' in residue else residue.get_atoms()
                            for atom in atoms:
                                if not self.ca_only and atom.element == 'H':
                                    continue
                                c = atom.get_coord()
                                coords.append([float(c[0]), float(c[1]), float(c[2])])
                                elem = atom.element.upper().strip()
                                a_num = 6 if elem == 'C' else 7 if elem == 'N' else 8 if elem == 'O' else 16 if elem == 'S' else 0
                                atomic_nums.append([a_num])
                    break
            except Exception as e:
                log_warn(f"Biopython fallback failed: {e}", stage="GNNParser")
                return None, f"Biopython fallback failed: {e}"

        if not coords:
            return None, "All protein parsing attempts failed"

        return self._coords_to_graph(coords, atomic_nums)

    def _coords_to_graph(self, coords: List[List[float]], atomic_nums: List[List[int]]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Convert 3D coordinates and atomic features into a graph dictionary.
        """
        coords_arr = np.array(coords)
        dist_mat = cdist(coords_arr, coords_arr)
        adj = np.where((dist_mat < self.dist_threshold) & (dist_mat > 0))
        edge_index = np.stack(adj).tolist()

        return {
            'x': atomic_nums,
            'pos': coords_arr.tolist(),
            'edge_index': edge_index,
        }, None

    def _process_ligand(self, mol: Chem.Mol) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Convert a ligand molecule into graph data with node and positional features.
        """
        try:
            if not mol:
                return None, "ligand_load_error"
            xs = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic())] for a in mol.GetAtoms()]
            pos = mol.GetConformer().GetPositions().tolist()
            edges = []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                edges.extend([[i, j], [j, i]])
            return {'x': xs, 'pos': pos, 'edge_index': edges}, None
        except Exception as e:
            log_warn(f"ligand processing failed: {e}", stage="GNNParser")
            return None, str(e)
