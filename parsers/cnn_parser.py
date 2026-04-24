# parsers/cnn_parser.py

from typing import Any, Optional, Tuple
from rdkit import Chem, RDLogger
from Bio.PDB import PDBParser, PPBuilder
from ._base_parser import BaseParser
from logger import log_debug, log_info, log_warn, log_error

RDLogger.DisableLog('rdApp.*')


class CNNParser(BaseParser):
    """
    Parser for extracting protein sequences or ligand SMILES suitable for CNN input.

    Uses RDKit and Biopython fallback logic to extract reliable sequence information.
    """

    def __init__(self, is_ligand: bool = False) -> None:
        """
        Initialize CNN parser.

        Args:
            is_ligand: Whether to parse ligand files instead of protein PDB files.
        """
        self.is_ligand = is_ligand
        log_info(f"Initialized is_ligand={is_ligand}", stage="CNNParser")

    def parse_file(self, path: str) -> Tuple[Optional[Any], Optional[str]]:
        """
        Parse a file into either a protein sequence or ligand SMILES.

        Args:
            path: Path to the input file.

        Returns:
            Tuple of parsed object and error message. Error message is None on success.
        """
        try:
            if self.is_ligand:
                mol = Chem.MolFromMolFile(path, sanitize=False)
                if not mol:
                    return None, "ligand_load_error"
                return self._process_ligand(mol)
            return self._process_protein(path)
        except Exception as e:
            log_warn(f"Parse_file failed: {e}", stage="CNNParser")
            return None, str(e)

    def _process_ligand(self, mol: Chem.Mol) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert ligand molecule into canonical SMILES.
        """
        try:
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass
            return Chem.MolToSmiles(mol, isomericSmiles=True), None
        except Exception as e:
            log_warn(f"Ligand processing failed: {e}", stage="CNNParser")
            return None, str(e)

    def _process_protein(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract an amino acid sequence from a protein PDB file.
        """
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        try:
            mol = Chem.MolFromPDBFile(path, sanitize=False, proximityBonding=False)
            if mol:
                seq = Chem.MolToSequence(mol)
                res_seq = "".join([res for res in seq if res in valid_aa])
                if res_seq:
                    return res_seq, None
        except Exception:
            pass

        try:
            parser = PDBParser(QUIET=True)
            ppb = PPBuilder()
            struct = parser.get_structure("prot", path)
            seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(struct))
            if seq:
                return seq, None
            return None, "empty_sequence_in_both_parsers"
        except Exception as e:
            log_warn(f"Protein fallback failed: {e}", stage="CNNParser")
            return None, f"biopython_fallback_error: {e}"
