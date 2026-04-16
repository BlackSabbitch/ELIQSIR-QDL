# parsers/interaction_graph_parser.py

import numpy as np
import io
from rdkit import Chem
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from ._base_parser import BaseParser

class InteractionGraphParser(BaseParser):
    """
    Создает единый 3D-граф взаимодействия (Лиганд + Карман).
    На выходе: словарь с 'x' (фичи узлов), 'pos' (координаты) и 'edge_index'.
    """
    def __init__(self, dist_threshold=5.0, ca_only=False):
        self.dist_threshold = dist_threshold
        self.ca_only = ca_only
        self.pdb_parser = PDBParser(QUIET=True)

    def _process_ligand(self):
        pass

    def _process_protein(self):
        pass

    def parse_file(self, lig_path, pock_path=None):
        # Этот парсер особенный: ему нужны два файла одновременно. 
        # Если пришел один (например, оркестратор вызывает старый метод), кидаем ошибку.
        if pock_path is None:
            return None, "EGNNParser требует lig_path и pock_path одновременно"
            
        try:
            return self._build_complex_graph(lig_path, pock_path, is_file=True)
        except Exception as e:
            return None, str(e)

    def parse_stream(self, lig_bytes, pock_bytes=None):
        if pock_bytes is None:
             return None, "EGNNParser требует lig_bytes и pock_bytes одновременно"
        try:
            return self._build_complex_graph(lig_bytes, pock_bytes, is_file=False)
        except Exception as e:
            return None, str(e)

    def _build_complex_graph(self, lig_data, pock_data, is_file=True):
        # 1. ПАРСИМ ЛИГАНД
        if is_file:
            lig_mol = Chem.MolFromMolFile(lig_data, sanitize=False)
        else:
            lig_content = lig_data.decode('utf-8') if isinstance(lig_data, bytes) else lig_data
            lig_mol = Chem.MolFromMolBlock(lig_content, sanitize=False)
            
        if not lig_mol: return None, "ligand_load_error"
        
        # Легкая санитаризация для фичей
        try: Chem.SanitizeMol(lig_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except: pass
        
        lig_coords = lig_mol.GetConformer().GetPositions()
        # Фичи: [AtomicNum, Degree, IsAromatic, 1 (индикатор лиганда)]
        lig_x = [[a.GetAtomicNum(), a.GetDegree(), int(a.GetIsAromatic()), 1] for a in lig_mol.GetAtoms()]

        # 2. ПАРСИМ КАРМАН
        pock_coords = []
        pock_x = []
        
        if is_file:
            struct = self.pdb_parser.get_structure("pock", pock_data)
        else:
            stream = io.StringIO(pock_data.decode('utf-8') if isinstance(pock_data, bytes) else pock_data)
            struct = self.pdb_parser.get_structure("pock", stream)

        # Классический обход Biopython
        for model in struct:
            for chain in model:
                for residue in chain:
                    # ФЛАГ РЕДУКЦИИ: CA_only или All-Heavy-Atoms
                    atoms_to_process = [residue['CA']] if self.ca_only and 'CA' in residue else residue.get_atoms()
                    
                    for atom in atoms_to_process:
                        if not self.ca_only and atom.element == 'H': continue # Без водорода
                        
                        c = atom.get_coord()
                        pock_coords.append([float(c[0]), float(c[1]), float(c[2])])
                        
                        # Фичи белка: [AtomicNum, 0, 0, 0 (индикатор белка)]
                        atomic_num = 6 if atom.element == 'C' else (7 if atom.element == 'N' else (8 if atom.element == 'O' else (16 if atom.element == 'S' else 0)))
                        pock_x.append([atomic_num, 0, 0, 0])
            break # Берем только первую модель (избегаем дублей из ЯМР)

        if not pock_coords: return None, "empty_pocket_ca" if self.ca_only else "empty_pocket"

        # 3. СШИВАЕМ И СОЗДАЕМ РЕБРА
        all_x = np.array(lig_x + pock_x)
        all_coords = np.concatenate([lig_coords, np.array(pock_coords)], axis=0)

        edges = []
        # Внутри лиганда
        for b in lig_mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            edges += [[i, j], [j, i]]
            
        # Между лигандом и карманом (по расстоянию)
        dist_mat = cdist(lig_coords, pock_coords)
        lig_idx, pock_idx = np.where(dist_mat < self.dist_threshold)
        
        for i, j in zip(lig_idx, pock_idx):
            p_idx_shifted = j + len(lig_x) # Сдвиг на число атомов лиганда
            edges += [[i, p_idx_shifted], [p_idx_shifted, i]]

        return {
            'x': all_x.tolist(), 
            'pos': all_coords.tolist(), 
            'edge_index': edges
        }, None
