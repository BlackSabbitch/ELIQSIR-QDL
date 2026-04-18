# parsers/cnn_parser.py

from rdkit import Chem, RDLogger
from Bio.PDB import PDBParser, PPBuilder
from ._base_parser import BaseParser

RDLogger.DisableLog('rdApp.*')


class CNNParser(BaseParser):
    def __init__(self, is_ligand=False):
        self.is_ligand = is_ligand
        self.valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")

    def parse_file(self, path):
        try:
            if self.is_ligand:
                mol = Chem.MolFromMolFile(path, sanitize=False)
                if not mol: return None, "ligand_load_error"
                return self._process_ligand(mol)
            else:
                return self._process_protein(path)
        except Exception as e:
            return None, str(e)

    def _process_ligand(self, mol):
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            # Возвращаем стереохимию и кекулизацию!
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except: 
                pass
                
            return Chem.MolToSmiles(mol, isomericSmiles=True), None
        except Exception as e:
            return None, str(e)

    def _process_protein(self, path):
        try:
            mol = Chem.MolFromPDBFile(path, sanitize=False, proximityBonding=False)
            
            if mol:
                seq = Chem.MolToSequence(mol)
                res_seq = "".join([res for res in seq if res in self.valid_aa])
                if res_seq: return res_seq, None
        except:
            pass

        # Если быстрый метод не сработал, пробуем медленный
        try:
            parser = PDBParser(QUIET=True)
            ppb = PPBuilder()
            struct = parser.get_structure("prot", path)
            
            seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(struct))

            if seq: return seq, None
            else: return None, "empty_sequence_in_both_parsers"
        except Exception as e:
            return None, f"biopython_fallback_error: {str(e)}"
