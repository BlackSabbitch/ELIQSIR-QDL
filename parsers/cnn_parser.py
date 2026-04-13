# parsers/cnn_parser.py

from rdkit import Chem, RDLogger
import io
from Bio.PDB import PDBParser, PPBuilder
from .base_parser import BaseParser

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

    def _process_protein(self, path_or_bytes, is_file=True):
        try:
            if is_file:
                mol = Chem.MolFromPDBFile(path_or_bytes, sanitize=False, proximityBonding=False)
            else:
                content = path_or_bytes.decode('utf-8')
                mol = Chem.MolFromPDBBlock(content, sanitize=False, proximityBonding=False)
            
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
            if is_file:
                struct = parser.get_structure("prot", path_or_bytes)
            else:
                stream = io.StringIO(path_or_bytes.decode('utf-8') if isinstance(path_or_bytes, bytes) else path_or_bytes)
                struct = parser.get_structure("prot", stream)
            
            seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(struct))

            if seq: return seq, None
            else: return None, "empty_sequence_in_both_parsers"
        except Exception as e:
            return None, f"biopython_fallback_error: {str(e)}"
