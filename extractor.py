# extractor.py

from datetime import datetime
import json
import os
import tarfile
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter


class PDBBindOrchestrator:
    def __init__(self, prot_parser, lig_parser, pock_parser, 
                 archive_path='pdbbind_v2016.tar.gz', dest_path='data'):
        self.prot_parser = prot_parser
        self.lig_parser = lig_parser
        self.pock_parser = pock_parser
        self.archive_path = archive_path
        self.dest_path = dest_path
        self.index_map = {
            "core": "v2016/index/INDEX_core_data.2016",
            "refined": "v2016/index/INDEX_refined_data.2016",
            "general": "v2016/index/INDEX_general_PL_data.2016"
        }

    def prepare_metadata(self):
        if os.path.exists(os.path.join(self.dest_path, "v2016/index")):
            return

        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for member in tar:
                if ('index' in member.name.lower() or 'readme' in member.name.lower()) and member.isfile():
                    tar.extract(member, path=self.dest_path)

    def get_complex_ids(self, subset="refined"):
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

    def extract_subset(self, subset="refined"):
        """Распаковывает выбранный набор на диск."""
        def is_safe_path(base, path):
            return os.path.realpath(path).startswith(os.path.realpath(base))

        targets = self.get_complex_ids(subset)
        print(f"Распаковка {len(targets)} комплексов...")
        
        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for member in tqdm(tar):
                parts = member.name.split('/')
                if len(parts) >= 2 and parts[1] in targets:
                    target_path = os.path.join(self.dest_path, member.name)
                    if not os.path.exists(target_path) and is_safe_path(self.dest_path, target_path):
                        tar.extract(member, path=self.dest_path)
        print("Распаковка завершена.")

    def _parse_single_complex(self, pdb_id):
        """Параллельный парсинг с диска."""
        folder = os.path.join(self.dest_path, "v2016", pdb_id)
        p_path = os.path.join(folder, f"{pdb_id}_protein.pdb")
        l_path = os.path.join(folder, f"{pdb_id}_ligand.sdf")
        pk_path = os.path.join(folder, f"{pdb_id}_pocket.pdb")

        if not all(os.path.exists(p) for p in [p_path, l_path, pk_path]):
            return (pdb_id, None, "missing_files")

        p_res, p_err = self.prot_parser.parse_file(p_path)
        if p_err: return (pdb_id, None, f"protein_parse_error: {p_err}")

        l_res, l_err = self.lig_parser.parse_file(l_path)
        if l_err: return (pdb_id, None, f"ligand_parse_error: {l_err}")

        pk_res, pk_err = self.pock_parser.parse_file(pk_path)
        if pk_err: return (pdb_id, None, f"pocket_parse_error: {pk_err}")
        
        return (pdb_id, {'seq': p_res, 'pocket_seq': pk_res, 'smiles': l_res}, None)

    def _build(self, ids, targets, n_jobs=-1):
        results, errors = [], []
        print(f"Запуск параллельного парсинга на {n_jobs if n_jobs > 0 else os.cpu_count()} ядрах...")
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

        print(f"Успешно: {len(results)}, Ошибок: {len(errors)}")
        print(Counter(errors))
        return pd.DataFrame(results)

    def build_dataset(self, subset="refined", use_disk=True, n_jobs=-1,
                      save_dir="datasets", file_name=None, fmt="parquet", compression="snappy"):
        """
        Собирает датасет и сохраняет его с расширенными опциями.
        save_dir: директория для сохранения
        file_name: имя файла (без расширения). По умолчанию: pdbbind_{subset}
        fmt: формат файла ('parquet', 'pickle'/'pkl', 'csv')
        compression: тип сжатия (snappy, gzip, brotli для parquet; в pickle зависит от библиотеки)
        """
        targets = self.get_complex_ids(subset)
        
        ids_on_disk = [pid for pid in targets.keys() if 
                       os.path.exists(os.path.join(self.dest_path, "v2016", pid))]
            
        df = self._build(ids_on_disk, targets, n_jobs)

        column_order = ['pdb_id', 'pkd', 'res', 'smiles', 'seq', 'pocket_seq']
        ordered_cols = [col for col in column_order if col in df.columns]
        df = df[ordered_cols]

        code = self._make_alias(subset)
        name = file_name if file_name else f"pdbbind_{code}"
        self.full_path = os.path.join(save_dir, f"{name}.{fmt}")
        actual_comp = compression if fmt == "parquet" else (None if compression == "snappy" else compression)

        os.makedirs(save_dir, exist_ok=True)
        self._save_metadata(subset, save_dir, name, fmt, actual_comp, len(df))
        self._save_dataset(df, fmt, actual_comp)

        return df

    def _make_alias(self, subset):
        prot_parser_name = self.prot_parser.__class__.__name__[0]
        if prot_parser_name == "GNN":
            prot_parser_name += f"{str(self.prot_parser.dist_threshold).replace('.', '_')}"
        lig_parser_name = self.lig_parser.__class__.__name__[0]
        if lig_parser_name == "GNN":
            lig_parser_name += f"{str(self.lig_parser.dist_threshold).replace('.', '_')}"
        pock_parser_name = self.pock_parser.__class__.__name__[0]
        if pock_parser_name == "GNN":
            pock_parser_name += f"{str(self.pock_parser.dist_threshold).replace('.', '_')}"
        return f"{subset}_prot{prot_parser_name}_lig{lig_parser_name}_pock{pock_parser_name}"


    def _save_dataset(self, df, fmt, actual_comp):
        if fmt == "parquet":
            df.to_parquet(self.full_path, compression=actual_comp)
        elif fmt in ["pickle", "pkl"]:
            df.to_pickle(self.full_path, compression=actual_comp)
        elif fmt == "csv":
            df.to_csv(self.full_path, index=False, compression=actual_comp)
            
        print(f"Датасет сохранен в {self.full_path} (сжатие: {actual_comp})")

    def _save_metadata(self, subset, save_dir, name, fmt, actual_comp, n_complexes):
        # 2. Собираем метаданные оркестратора и парсеров
        metadata = {
            "full_path": self.full_path,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "subset": subset,
            "format": fmt,
            "compression": str(actual_comp),
            "n_complexes": n_complexes,
            # Если у твоих парсеров есть атрибуты (например is_ligand), их тоже можно вытащить:
            "parsers": {
                "protein": self.prot_parser.__class__.__name__,
                "ligand": self.lig_parser.__class__.__name__,
                "pocket": self.pock_parser.__class__.__name__
            }
        }
        
        # 3. Сохраняем JSON рядом с датасетом
        self.full_meta_path = os.path.join(save_dir, f"{name}_meta.json")
        with open(self.full_meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)

        print(f"Метаданные сохранены в {self.full_meta_path}")
