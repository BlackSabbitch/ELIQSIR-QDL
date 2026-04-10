# extractor.py

import os
import tarfile
import io
import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from multiprocessing import Pool
from collections import Counter

# Отключаем предупреждения RDKit для чистоты вывода
RDLogger.DisableLog('rdApp.*')

class PDBBindProcessor:
    def __init__(self, archive_path='pdbbind_v2016.tar.gz', dest_path='data'):
        self.archive_path = archive_path
        self.dest_path = dest_path
        self.index_map = {
            "core": "v2016/index/INDEX_core_data.2016",
            "refined": "v2016/index/INDEX_refined_data.2016",
            "general": "v2016/index/INDEX_general_PL_data.2016"
        }
        # Парсеры инициализируем здесь, но внутри процессов создадим новые (они не всегда pickle-able)
        self.parser = PDBParser(QUIET=True)
        self.ppb = PPBuilder()

    def prepare_metadata(self):
        """Извлекает индексы и readme."""
        if os.path.exists(os.path.join(self.dest_path, "v2016/index")):
            return
        print("Extracting metadata...")
        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for member in tar:
                if ('index' in member.name.lower() or 'readme' in member.name.lower()) and member.isfile():
                    tar.extract(member, path=self.dest_path)

    def get_complex_ids(self, subset="refined"):
        """Парсит индекс и возвращает словарь с метаданными."""
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
        """Рабочая функция для одного процесса."""
        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()
        
        folder = os.path.join(self.dest_path, "v2016", pdb_id)
        prot_path = os.path.join(folder, f"{pdb_id}_protein.pdb")
        lig_path = os.path.join(folder, f"{pdb_id}_ligand.sdf")
        pocket_path = os.path.join(folder, f"{pdb_id}_pocket.pdb")

        if not (os.path.exists(prot_path) and os.path.exists(lig_path) and os.path.exists(pocket_path)):
            return (pdb_id, None, None, None, "missing_files")

        try:
            # 1. Обработка белка
            struct = parser.get_structure(pdb_id, prot_path)
            seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(struct))
            
            # 2. Карман (новая часть)
            pock_struct = parser.get_structure(pdb_id, pocket_path)
            pock_seq = "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(pock_struct))

            if not seq or not pock_seq: return (pdb_id, None, None, None, "empty_sequence")
            if len(seq) > 2000: return (pdb_id, None, None, None, "too_long")

            # 2. Обработка лиганда
            mol = Chem.MolFromMolFile(lig_path, sanitize=False)
            if not mol: return (pdb_id, None, None, None, "rdkit_load_error")
            
            # Санитаризация
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                             Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except: pass
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            return (pdb_id, seq, pock_seq, smi, None)
        except Exception as e:
            return (pdb_id, None, None, None, str(e))

    def build_dataset(self, subset="refined", use_disk=True, n_jobs=-1, save_path=None):
        targets = self.get_complex_ids(subset)
        
        if use_disk:
            ids_on_disk = [pid for pid in targets.keys() if 
                           os.path.exists(os.path.join(self.dest_path, "v2016", pid))]
            
            if len(ids_on_disk) < len(targets):
                print(f"Предупреждение: Найдено только {len(ids_on_disk)} папок на диске.")
            
            df = self._build_parallel(ids_on_disk, targets, n_jobs)
        else:
            df = self._build_streaming(targets)

        if save_path:
            df.to_parquet(save_path)
            print(f"Датасет сохранен в {save_path}")
        return df

    def _build_parallel(self, ids, targets, n_jobs):
        if n_jobs == -1: n_jobs = os.cpu_count()
        print(f"Запуск параллельного парсинга на {n_jobs} ядрах...")
        
        results = []
        errors = []
        with Pool(n_jobs) as pool:
            for res in tqdm(pool.imap(self._parse_single_complex, ids, chunksize=16), total=len(ids)):
                if res:
                    pdb_id, seq, pock_seq, smi, err = res
                    if seq and pock_seq and smi:
                        results.append({
                            'pdb_id': pdb_id, 
                            'pkd': targets[pdb_id]['pkd'],
                            'res': targets[pdb_id]['res'], 
                            'smiles': smi, 
                            'seq': seq,
                            'pocket_seq': pock_seq
                        })
                    else:
                        errors.append((pdb_id, err))
        
        print(f"Успешно: {len(results)}, Ошибок: {len(errors)}")
        error_types = Counter(err for _, err in errors)
        print(error_types)
        return pd.DataFrame(results)

    def _build_streaming(self, targets):
        """Режим работы напрямую с архивом (без распаковки)."""
        results = []
        pending_ids = {pdb_id: {} for pdb_id in targets}
        
        print("Потоковое чтение из архива...")
        with tarfile.open(self.archive_path, 'r:gz') as tar:
            for member in tqdm(tar):
                if not pending_ids: break
                if not member.isfile(): continue
                
                parts = member.name.split('/')
                if len(parts) < 3: continue
                
                pdb_id = parts[1]
                if pdb_id not in pending_ids: continue
                
                f_obj = tar.extractfile(member)
                content = f_obj.read()
                
                if parts[2].endswith('protein.pdb'):
                    pending_ids[pdb_id]['seq'] = self._get_seq_from_stream(content, pdb_id)
                elif parts[2].endswith('pocket.pdb'):
                    pending_ids[pdb_id]['pocket_seq'] = self._get_seq_from_stream(content, pdb_id)
                elif parts[2].endswith('ligand.sdf'):
                    pending_ids[pdb_id]['smi'] = self._get_smi_from_stream(content)

                if all(key in pending_ids[pdb_id] for key in ['seq', 'pocket_seq', 'smi']):
                    s, p, m = pending_ids[pdb_id]['seq'], pending_ids[pdb_id]['pocket_seq'], pending_ids[pdb_id]['smi']
                    if s and p and m:
                        results.append({
                            'pdb_id': pdb_id, 'pkd': targets[pdb_id]['pkd'],
                            'res': targets[pdb_id]['res'], 'smiles': m,
                            'seq': s, 'pocket_seq': p
                        })
                    del pending_ids[pdb_id]
        
        return pd.DataFrame(results)

    def _get_seq_from_stream(self, binary_content, pdb_id):
        try:
            stream = io.StringIO(binary_content.decode('utf-8'))
            struct = self.parser.get_structure(pdb_id, stream)
            seq = "".join(str(pp.get_sequence()) for pp in self.ppb.build_peptides(struct))
            return seq if (seq and len(seq) <= 2000) else None
        except Exception:
            return None

    def _get_smi_from_stream(self, binary_content):
        try:
            text = binary_content.decode('utf-8')
            mol = Chem.MolFromMolBlock(text, sanitize=False)
            if not mol:
                suppl = Chem.ForwardSDMolSupplier(io.BytesIO(binary_content), sanitize=False)
                mol = next(suppl)
            
            if mol:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ 
                                                 Chem.SanitizeFlags.SANITIZE_PROPERTIES)
                Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
                return Chem.MolToSmiles(mol, isomericSmiles=True)
        except Exception:
            return None
