# parsers/gnn_parser.py

import numpy as np
import io
from rdkit import Chem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from .base_parser import BaseParser

class GNNParser(BaseParser):
    def __init__(self, is_ligand=False, dist_threshold=10.0):
        self.is_ligand = is_ligand
        self.dist_threshold = dist_threshold

    def parse_file(self, path):
        try:
            if self.is_ligand:
                mol = Chem.MolFromMolFile(path, sanitize=False)
                if not mol: return None, "ligand_load_error"
                return self._process_ligand(mol)
            else:
                return self._process_protein(path, is_file=True)
        except Exception as e:
            return None, str(e)

    def parse_stream(self, binary_content):
        try:
            if self.is_ligand:
                content = binary_content.decode('utf-8')
                mol = Chem.MolFromMolBlock(content, sanitize=False)
                if not mol:
                    suppl = Chem.ForwardSDMolSupplier(io.BytesIO(binary_content), sanitize=False)
                    try: mol = next(suppl)
                    except: mol = None
                if not mol: return None, "ligand_load_error"
                return self._process_ligand(mol)
            else:
                return self._process_protein(binary_content, is_file=False)
        except Exception as e:
            return None, str(e)

    def _process_ligand(self, mol):
        try:
            # Легкая санитаризация для вычисления валентности и ароматики
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            except: pass
            
            xs = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic())] for a in mol.GetAtoms()]
            edges = []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                # Граф неориентированный, добавляем связи в обе стороны
                edges += [[i, j], [j, i]]
                
            return {'x': xs, 'edge_index': edges}, None
        except Exception as e:
            return None, f"ligand_graph_error: {str(e)}"

    def _coords_to_graph(self, coords):
        """Вспомогательный метод для перевода координат в граф"""
        coords = np.array(coords)
        dist_mat = cdist(coords, coords)
        # Находим индексы атомов, расстояние между которыми меньше порога (и > 0, чтобы исключить сам атом)
        adj = np.where((dist_mat < self.dist_threshold) & (dist_mat > 0))
        return {'x': coords.tolist(), 'edge_index': np.stack(adj).tolist()}, None

    def _process_protein(self, path_or_bytes, is_file=True):
        # ПОПЫТКА 1: Быстрый RDKit
        try:
            if is_file:
                mol = Chem.MolFromPDBFile(path_or_bytes, sanitize=False, proximityBonding=False)
            else:
                content = path_or_bytes.decode('utf-8') if isinstance(path_or_bytes, bytes) else path_or_bytes
                mol = Chem.MolFromPDBBlock(content, sanitize=False, proximityBonding=False)

            if mol and mol.GetNumConformers() > 0:
                coords = []
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    info = atom.GetPDBResidueInfo()
                    if info and info.GetName().strip() == "CA":
                        pos = conf.GetAtomPosition(atom.GetIdx())
                        coords.append([pos.x, pos.y, pos.z])
                if coords:
                    return self._coords_to_graph(coords)
        except:
            pass # Если RDKit упал, тихо переходим к Biopython

        # ПОПЫТКА 2: Надежный Biopython (Архитектурный подход)
        try:
            parser = PDBParser(QUIET=True)
            if is_file:
                struct = parser.get_structure("prot", path_or_bytes)
            else:
                stream = io.StringIO(path_or_bytes.decode('utf-8') if isinstance(path_or_bytes, bytes) else path_or_bytes)
                struct = parser.get_structure("prot", stream)

            coords = []
            for model in struct:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            # Вытаскиваем X, Y, Z у альфа-углерода
                            c = residue['CA'].get_coord()
                            coords.append([float(c[0]), float(c[1]), float(c[2])])
                break # Берем только первую модель (Model 0), чтобы избежать дублей из NMR

            if coords:
                return self._coords_to_graph(coords)
            return None, "empty_ca_coordinates_in_both_parsers"
            
        except Exception as e:
            return None, f"biopython_fallback_error: {str(e)}"
