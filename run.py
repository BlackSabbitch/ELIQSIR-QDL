# run.py

import json
import os
import numpy as np
import pandas as pd
import gc
import argparse
import tempfile
import shutil
import torch
from datetime import datetime
from torch_geometric.loader import DataLoader
from typing import Tuple

from logger import *
from extractor import PDBBindOrchestrator
from tokenizer import UniversalPDBBindDataset
from evaluator import Evaluator
from splitter import PDBBindSplitter
from model.model_builder import UHSMBuilder
from parsers.cnn_parser import CNNParser
from parsers.gnn_parser import GNNParser
from model.trainer import HybridTrainer
from utils import Utils


DATASETS_DIR = "datasets"


class ExperimentRunner:
    """
    Orchestrates the full pipeline: data extraction, parsing, splitting,
    training, and evaluation based on a configuration file.
    """

    def __init__(self, config_path: str, extract: bool = False,
                 train_dataset_path: None | str = None,
                 test_dataset_path: None |str = None,
                 val_dataset_path: None | str = None,
                 temp_run: bool = False,
                 keep_temp: bool = False,
                 exp_dir: None | str = None):
        with open(config_path or 'config.json', 'r') as f:
            self.config = json.load(f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log_info(f"Starting experiment: {self.config['experiment_name']}", stage="EXPERIMENT")
        self.extract = extract
        self.save_train_test_val_datasets = self.config['dataset']['save_train_test_val_datasets']
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.val_dataset_path = val_dataset_path
        self.temp_run = temp_run
        self.keep_temp = keep_temp
        self.exp_dir = exp_dir
        assert all([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path]) or not any([self.train_dataset_path, self.test_dataset_path, self.val_dataset_path])

    def prepare_folders(self):
        if self.exp_dir is not None:
            self.exp_run_dir = self.exp_dir
        elif self.temp_run:
            os.makedirs('runs', exist_ok=True)
            self.exp_run_dir = tempfile.mkdtemp(prefix=f"{self.config['experiment_name']}_tmp_", dir="runs")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"{self.config['experiment_name']}_{timestamp}"
            log_info(f"Experiment signature: {exp_name}", stage="EXPERIMENT")
            self.exp_run_dir = f"runs/{exp_name}"

        os.makedirs(DATASETS_DIR, exist_ok=True)
        log_info(f"Base Datasets folder: {DATASETS_DIR}", stage="EXPERIMENT")

        os.makedirs(self.exp_run_dir, exist_ok=True)
        log_info(f"Run results folder: {self.exp_run_dir}", stage="EXPERIMENT")

        if self.save_train_test_val_datasets:
            self.exp_run_datasets_dir = f"{self.exp_run_dir}/datasets"
            os.makedirs(self.exp_run_datasets_dir, exist_ok=True)
            log_info(f"Experiment datasets path: {self.exp_run_datasets_dir}", stage="EXPERIMENT")

        log_path = os.path.join(self.exp_run_dir, "log.txt")
        setup_file_logging(log_path)
        log_info(f"Log file: {log_path}", stage="EXPERIMENT")

    def _get_dataset_from_path(self, path: str):
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith(('.pkl', '.pickle')):
            return pd.read_pickle(path)
        elif path.endswith('.parquet'):
            return pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file format. Use .csv, .pkl, or .parquet")

    def prepare_datasets(self):
        if self.train_dataset_path is not None:
            log_info(f"Run with custom train/test/val datasets", stage="EXPERIMENT")

            self.config["dataset"].update({
                "train_path": self.train_dataset_path,
                "test_path": self.test_dataset_path,
                "val_path": self.val_dataset_path,
                })
            train_df = self._get_dataset_from_path(self.train_dataset_path)
            test_df = self._get_dataset_from_path(self.test_dataset_path)
            val_df = self._get_dataset_from_path(self.val_dataset_path)

        else:
            mode = self.config['model']['graph_encoder']['selected']
            mode_str = self.config['model']['graph_encoder']['available'][mode]['protein_ligand_pocket_encoders']
            parsers = []
            for i, char in enumerate(mode_str):
                is_lig = (i == 1)
                if char == 'C': parsers.append(CNNParser(is_ligand=is_lig))
                elif char == 'G': parsers.append(GNNParser(is_ligand=is_lig))
                elif char == 'N': parsers.append(None)
                else: log_info(f"Unknown symbol {char} in the architecture description.", stage="EXPERIMENT")

            orchestrator = PDBBindOrchestrator(parsers, self.config)
            if self.extract: orchestrator.extract_subset("refined")
            df_refined = orchestrator.build_dataset(subset="refined", fmt="pickle", save_dir=DATASETS_DIR)
            df_core = orchestrator.build_dataset(subset="core", fmt="pickle", save_dir=DATASETS_DIR)

            clean_refined = df_refined[~df_refined['pdb_id'].isin(df_core['pdb_id'])]

            train_df, val_df = PDBBindSplitter.split(clean_refined, self.config["splitter"])

            test_df = df_core.copy()

            if self.save_train_test_val_datasets:
                train_path = f"{self.exp_run_datasets_dir}/train.pickle"
                val_path   = f"{self.exp_run_datasets_dir}/val.pickle"
                test_path  = f"{self.exp_run_datasets_dir}/test_core.pickle"

                train_df.to_pickle(train_path)
                test_df.to_pickle(test_path)
                val_df.to_pickle(val_path)

                self.config["dataset"].update({
                    "train_path": train_path,
                    "test_path": test_path,
                    "val_path": val_path,
                    })

        # Normalization
        log_info(f"Data normalization", stage="EXPERIMENT")
        train_target = train_df['pkd'].values
        stats = {'mean': float(train_target.mean()), 'std': float(train_target.std())}
        self.config['dataset']['stats'] = stats
        for df in [train_df, val_df, test_df]:
            df['pkd'] = Utils.normalize(df['pkd'].values, stats)
        log_info(f"Data Stats: {stats}", stage="EXPERIMENT")
        log_info(f"min value: {train_df['pkd'].min()}, max value: {train_df['pkd'].max()}", stage="EXPERIMENT")

        return train_df, val_df, test_df

    def run(self, train_df, val_df, test_df):
        train_ds = UniversalPDBBindDataset(train_df, self.config)
        test_ds = UniversalPDBBindDataset(test_df, self.config)
        val_ds   = UniversalPDBBindDataset(val_df, self.config)
        gc.collect()

        train_loader = DataLoader(train_ds, batch_size=self.config['dataset']['batch_size'], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self.config['dataset']['batch_size'], shuffle=False)
        val_loader   = DataLoader(val_ds, batch_size=self.config['dataset']['batch_size'], shuffle=False)

        self.config['dataset']['actual_sizes'] = {
            'train': len(train_ds),
            'val': len(val_ds),
            'test': len(test_ds)
        }

        with open(f"{self.exp_run_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

        model = UHSMBuilder.build_model_from_config(self.config)
        evaluator = Evaluator(model, self.device)

        log_info(f"Launch on: {self.device}", stage="EXPERIMENT")

        self.trainer = HybridTrainer(model, evaluator, self.config, self.device)
        best_epoch, _ = self.trainer.train(train_loader, val_loader, self.exp_run_dir, self.config['training']['save_only_best_epoch'])
        self.trainer.test(test_loader, self.exp_run_dir, best_epoch)

        log_info("Generating ASCII performance summary...", stage="SUMMARY")
        console_plots(self.trainer.history, side_by_side=False, stage="SUMMARY")
        console_plots(self.trainer.history, side_by_side=True, stage="SUMMARY")
        log_info("Experiment completed successfully.", stage="EXPERIMENT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDBBind Experiment")

    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file')
    
    # Флаг экстракции (если указан в bash — станет True)
    parser.add_argument('--extract', action='store_true', 
                        help='Extract subset before building dataset')
    
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--val_path', type=str, default=None)
    parser.add_argument('--temp-run', action='store_true', help='Use a temporary experiment directory and remove it after successful completion')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary directory even after successful temp-run')
    parser.add_argument('--exp-dir', type=str, default=None, help='Use a fixed experiment directory instead of timestamped runs/<name>_<ts>')

    args = parser.parse_args()
    runner = ExperimentRunner(
        config_path=args.config,
        extract=args.extract,
        train_dataset_path=args.train_path,
        test_dataset_path=args.test_path,
        val_dataset_path=args.val_path,
        temp_run=args.temp_run,
        keep_temp=args.keep_temp,
        exp_dir=args.exp_dir
    )
    runner.prepare_folders()
    train_df, val_df, test_df = runner.prepare_datasets()

    try:
        runner.run(train_df, val_df, test_df)
        if runner.temp_run and not runner.keep_temp:
            shutil.rmtree(runner.exp_run_dir, ignore_errors=True)
    except Exception as e:
        import traceback
        err_path = os.path.join(runner.exp_run_dir, "err_log.txt")
        error_msg = traceback.format_exc()
        log_info(f"ERROR message saved to: {err_path}", stage="CRASH")

        with open(err_path, "w", encoding="utf-8") as f:
            f.write(error_msg)

        raise e

"""
python run.py --config configs/gnn_test.json    # path to the custom config
python run.py --config config.json --extract    # then extract == True

"""
