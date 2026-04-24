# run.py

import json
import os
import numpy as np
import argparse
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

DATASETS_DIR = "datasets"


class ExperimentRunner:
    """
    Orchestrates the full pipeline: data extraction, parsing, splitting,
    training, and evaluation based on a configuration file.
    """

    def __init__(self, config_path: str, extract: bool = False,
                 train_dataset_path: None | str = None,
                 test_dataset_path: None |str = None,
                 val_dataset_path: None | str = None):
        with open(config_path or 'config.json', 'r') as f:
            self.config = json.load(f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log_info(f"Starting experiment: {self.config['experiment_name']}", stage="EXPERIMENT")
        self.extract = extract
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.val_dataset_path = val_dataset_path
        input_datasets = [self.train_dataset_path, self.test_dataset_path, self.val_dataset_path]
        assert all(input_datasets) or not any(input_datasets)

    def prepare_folders(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config['experiment_name']}_{timestamp}"        
        log_info(f"Experiment signature: {exp_name}", stage="EXPERIMENT")
        self.exp_run_dir = f"runs/{exp_name}"
        self.exp_run_datasets_dir = f"{self.exp_run_dir}/datasets"
        log_info(f"Base Datasets folder: {DATASETS_DIR}", stage="EXPERIMENT")
        log_info(f"Run results folder: {self.exp_run_dir}", stage="EXPERIMENT")
        log_info(f"Experiment datasets path: {self.exp_run_dir}/datasets", stage="EXPERIMENT")

        os.makedirs(self.exp_run_dir, exist_ok=True)
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(self.exp_run_datasets_dir, exist_ok=True)

        log_path = os.path.join(self.exp_run_dir, "log.txt")
        setup_file_logging(log_path)
        log_info(f"Log file: {log_path}", stage="EXPERIMENT")

    def prepare_datasets(self):
        if self.train_dataset_path is not None:
            log_info(f"Run with custom train/test/val datasets", stage="EXPERIMENT")

            self.config["dataset"].update({
                "train_path": self.train_dataset_path,
                "test_path": self.test_dataset_path,
                "val_path": self.val_dataset_path,
                })
            return

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

        test_df = df_core

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

    def run(self):
        train_ds = UniversalPDBBindDataset(self.config["dataset"]["train_path"], self.config)
        test_ds = UniversalPDBBindDataset(self.config["dataset"]["test_path"], self.config)
        val_ds   = UniversalPDBBindDataset(self.config["dataset"]["val_path"], self.config)

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

    args = parser.parse_args()
    runner = ExperimentRunner(
        config_path=args.config,
        extract=args.extract,
        train_dataset_path=args.train_path,
        test_dataset_path=args.test_path,
        val_dataset_path=args.val_path
    )
    runner.prepare_folders()
    runner.prepare_datasets()

    try:
        runner.run()
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
