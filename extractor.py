# extractor.py

from datetime import datetime
import json
import os
import tarfile
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple
from logger import logger


class PDBBindOrchestrator:
    """
    Orchestrator for extracting and processing PDBBind dataset complexes.

    Handles dataset preparation, file extraction, and parallel processing of
    protein-ligand complexes. Supports different dataset subsets (core, refined, general)
    and configurable parsers for molecular data processing.

    Attributes:
        parsers: List of parsers for processing different molecular components.
        mode: Selected encoder mode ('trio' or 'duo').
        archive_path: Path to the PDBBind archive file.
        dest_path: Destination directory for extracted data.
        index_map: Mapping of subset names to index file paths.
        file_suffix_map: Mapping of component types to file suffixes.

    Example:
        >>> from parsers.cnn_parser import CNNParser
        >>> from parsers.gnn_parser import GNNParser
        >>> parsers = [CNNParser(is_ligand=False), CNNParser(is_ligand=True), GNNParser(is_ligand=False)]
        >>> orchestrator = PDBBindOrchestrator(parsers, config)
        >>> orchestrator.extract_subset("refined")
        >>> df = orchestrator.build_dataset("refined")
    """

    def __init__(self, parsers: List[Any], config_dict: Dict[str, Any], archive_path: str = 'pdbbind_v2016.tar.gz', dest_path: str = 'data') -> None:
        """
        Initialize the PDBBind orchestrator.

        Args:
            parsers: List of parser objects for processing molecular data.
            config_dict: Configuration dictionary containing model settings.
            archive_path: Path to the PDBBind dataset archive.
            dest_path: Directory where data will be extracted.
        """
        self.parsers = parsers
        self.mode = config_dict['model']['graph_encoder']['selected']
        self.archive_path = archive_path
        self.dest_path = dest_path
        self.index_map = {
            "core": "v2016/index/INDEX_core_data.2016",
            "refined": "v2016/index/INDEX_refined_data.2016",
            "general": "v2016/index/INDEX_general_PL_data.2016"
            }
        self.file_suffix_map = {
            "protein": "_protein.pdb",
            "ligand": "_ligand.sdf",
            "pocket": "_pocket.pdb"
            }

    def prepare_metadata(self) -> None:
        """
        Prepare metadata by extracting index and readme files from archive.

        Checks if metadata already exists and extracts necessary files
        (index files and readme) from the PDBBind archive if not present.
        """
        if os.path.exists(os.path.join(self.dest_path, "v2016/index")):
            return

        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for member in tar:
                low_name = member.name.lower()
                if ('index' in low_name or 'readme' in low_name) and member.isfile():
                    tar.extract(member, path=self.dest_path)

    def _resolve_paths(self, pdb_id: str) -> List[Optional[str]]:
        """
        Resolve file paths for a given PDB complex ID.

        Args:
            pdb_id: PDB complex identifier.

        Returns:
            List of file paths for protein, ligand, and pocket files.
            None values indicate components that should be skipped.
        """
        folder = os.path.join(self.dest_path, "v2016", pdb_id)
        
        if len(self.parsers) == 3:
            # TRIO: account for parsers that might be None (mode 'N')
            return [
                os.path.join(folder, f"{pdb_id}{self.file_suffix_map['protein']}") if self.parsers[0] else None,
                os.path.join(folder, f"{pdb_id}{self.file_suffix_map['ligand']}") if self.parsers[1] else None,
                os.path.join(folder, f"{pdb_id}{self.file_suffix_map['pocket']}") if self.parsers[2] else None
            ]
        elif len(self.parsers) == 2:
            # EGNN / Interaction (Duo)
            return [
                os.path.join(folder, f"{pdb_id}{self.file_suffix_map['protein']}") if self.parsers[0] else None,
                os.path.join(folder, f"{pdb_id}{self.file_suffix_map['ligand']}"),
                os.path.join(folder, f"{pdb_id}{self.file_suffix_map['pocket']}")
            ]
        return []

    def get_complex_ids(self, subset: str = "refined") -> Dict[str, Dict[str, float]]:
        """
        Get complex IDs and affinity data for a dataset subset.

        Parses the index file for the specified subset and extracts valid
        complexes with resolution <= 2.5Å and positive affinity values.

        Args:
            subset: Dataset subset ('core', 'refined', or 'general').

        Returns:
            Dictionary mapping PDB IDs to affinity data (resolution and pKd).

        Raises:
            FileNotFoundError: If index file is not found.
        """
        self.prepare_metadata()
        index_path = os.path.join(self.dest_path, self.index_map[subset])
        affinity_table = {}
        with open(index_path, 'r') as f:
            for line in f:
                if not line or line.startswith('#'): continue
                parts = line.split()
                try:
                    pdb_id, res, pkd = parts[0], float(parts[1]), float(parts[3])
                    if res <= 2.5 and pkd > 0:
                        affinity_table[pdb_id] = {"res": res, "pkd": pkd}
                except (ValueError, IndexError): continue
        return affinity_table

    def extract_subset(self, subset: str = "refined") -> None:
        """
        Extract selected dataset subset to disk.

        Unpacks all complexes belonging to the specified subset from the archive.
        Only extracts complexes that pass quality filters (resolution, affinity).

        Args:
            subset: Dataset subset to extract ('core', 'refined', or 'general').

        Note:
            This operation may take significant time for large subsets.
        """
        def is_safe_path(base: str, path: str) -> bool:
            return os.path.realpath(path).startswith(os.path.realpath(base))

        targets = self.get_complex_ids(subset)
        logger.info(f"[EXTRACTION] Unpacking {len(targets)} complexes...")
        
        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for member in tqdm(tar):
                parts = member.name.split('/')
                if len(parts) >= 2 and parts[1] in targets:
                    target_path = os.path.join(self.dest_path, member.name)
                    if not os.path.exists(target_path) and is_safe_path(self.dest_path, target_path):
                        tar.extract(member, path=self.dest_path)
        logger.info("[EXTRACTION] Unpacking completed.")

    def _parse_single_complex(self, pdb_id: str) -> Tuple[str, Optional[Any], Optional[str]]:
        """
        Parse a single complex from disk files.

        Processes protein, ligand, and pocket files for a given PDB ID using
        the configured parsers. Used for parallel processing.

        Args:
            pdb_id: PDB complex identifier.

        Returns:
            Tuple of (pdb_id, parsed_data, error_message).
            parsed_data is None if parsing failed, error_message contains failure reason.
        """
        paths = self._resolve_paths(pdb_id)
        for p in paths:
            if p is not None and not os.path.exists(p):
                return (pdb_id, None, f"missing_file: {os.path.basename(p)}")

        data_results = {'protein': None, 'ligand': None, 'pocket': None, 'complex_graph': None}
        try:
            if len(self.parsers) == 3:
                # mode Trio (CCC, CGC, etc.)
                for i, parser, path in zip(range(3), self.parsers, paths):
                    if path is None: continue

                    res, err = parser.parse_file(path)
                    name_of_object = list(self.file_suffix_map.keys())[i] # protein / ligand / pocket
                    if err: return (pdb_id, None, f"{name_of_object}_parse_error: {err}")
                    data_results[name_of_object] = res

            elif len(self.parsers) == 2:
                # mode Duo (NE, CE, GE)
                protein_parser, complex_parser = self.parsers
                protein_path, ligand_path, pocket_path = paths

                if protein_parser:
                    p_res, p_err = protein_parser.parse_file(protein_path)
                    if p_err: return (pdb_id, None, f"protein_parse_error: {p_err}")
                    data_results['protein'] = p_res

                complex_res, complex_err = complex_parser.parse_file(ligand_path, pocket_path)
                if complex_err: return (pdb_id, None, f"complex_parse_error: {complex_err}")
                data_results['complex_graph'] = complex_res        
            return (pdb_id, data_results, None)
        except Exception as e:
            return (pdb_id, None, f"exception: {str(e)}")

    def _build(self, ids: List[str], targets: Dict[str, Dict[str, float]], n_jobs: int = -1) -> pd.DataFrame:
        """
        Build dataset using parallel processing.

        Args:
            ids: List of PDB IDs to process.
            targets: Dictionary of target affinity data.
            n_jobs: Number of parallel jobs (-1 for all cores).

        Returns:
            DataFrame containing parsed molecular data and affinity labels.
        """
        results, errors = [], []
        logger.info(f"[BUILD] Starting parallel parsing on {n_jobs if n_jobs > 0 else os.cpu_count()} cores...")
        with Pool(n_jobs if n_jobs > 0 else os.cpu_count()) as pool:
            for pid, data, err in tqdm(pool.imap(self._parse_single_complex, ids), total=len(ids)):
                if data:
                    data.update(
                        {'pdb_id': pid,
                         'pkd': targets[pid]['pkd'],
                         'res': targets[pid]['res']})
                    results.append(data)
                else:
                    errors.append(err)

        logger.info(f"[BUILD] Success: {len(results)}, Errors: {len(errors)}")
        if errors:
            logger.warning(f"[BUILD ERRORS] {Counter(errors)}")
        return pd.DataFrame(results)

    def build_dataset(self, subset: str = "refined", n_jobs: int = -1,
                      save_dir: str = "datasets", file_name: Optional[str] = None, 
                      fmt: str = "parquet", compression: str = "snappy") -> pd.DataFrame:
        """
        Build and save dataset with extended options.

        Processes all complexes in the specified subset, parses molecular data,
        and saves the resulting dataset in the requested format.

        Args:
            subset: Dataset subset ('core', 'refined', or 'general').
            n_jobs: Number of parallel jobs for processing.
            save_dir: Directory for saving the dataset.
            file_name: Output filename (without extension). Defaults to pdbbind_{subset}.
            fmt: File format ('parquet', 'pickle'/'pkl', 'csv').
            compression: Compression type (snappy, gzip, brotli for parquet).

        Returns:
            DataFrame containing the processed dataset.

        Example:
            >>> df = orchestrator.build_dataset("refined", n_jobs=4, fmt="parquet")
        """
        targets = self.get_complex_ids(subset)
        
        ids_on_disk = [pid for pid in targets.keys() if 
                       os.path.exists(os.path.join(self.dest_path, "v2016", pid))]
            
        df = self._build(ids_on_disk, targets, n_jobs)

        ideal_order = ['pdb_id', 'pkd', 'res', 'ligand', 'protein', 'pocket', 'complex_graph']
        existing_cols = [col for col in ideal_order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in existing_cols]
        df = df[existing_cols + remaining_cols]

        name = file_name if file_name else f"pdbbind_{self.mode}"
        self.full_path = os.path.join(save_dir, f"{name}.{fmt}")
        actual_comp = compression if fmt == "parquet" else (None if compression == "snappy" else compression)

        os.makedirs(save_dir, exist_ok=True)
        self._save_metadata(subset, save_dir, name, fmt, actual_comp, len(df))
        self._save_dataset(df, fmt, actual_comp)

        return df

    def _save_dataset(self, df: pd.DataFrame, fmt: str, actual_comp: Optional[str]) -> None:
        """
        Save dataset to file in specified format.

        Args:
            df: DataFrame to save.
            fmt: File format.
            actual_comp: Compression type.
        """
        if fmt == "parquet":
            df.to_parquet(self.full_path, compression=actual_comp)
        elif fmt in ["pickle", "pkl"]:
            df.to_pickle(self.full_path, compression=actual_comp)
        elif fmt == "csv":
            df.to_csv(self.full_path, index=False, compression=actual_comp)
            
        logger.info(f"[SAVE] Dataset saved to {self.full_path} (compression: {actual_comp})")

    def _save_metadata(self, subset: str, save_dir: str, name: str, fmt: str, actual_comp: Optional[str], n_complexes: int) -> None:
        """
        Save dataset metadata to JSON file.

        Args:
            subset: Dataset subset name.
            save_dir: Save directory.
            name: Dataset name.
            fmt: File format.
            actual_comp: Compression type.
            n_complexes: Number of complexes in dataset.
        """
        # 2. Collect orchestrator and parsers metadata
        metadata = {
            "full_path": self.full_path,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "subset": subset,
            "format": fmt,
            "compression": str(actual_comp),
            "n_complexes": n_complexes,
            "parsers": self.mode
        }
        
        # 3. Save JSON alongside dataset
        self.full_meta_path = os.path.join(save_dir, f"{name}_meta.json")
        with open(self.full_meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        logger.info(f"[SAVE] Metadata saved to {self.full_meta_path}")
