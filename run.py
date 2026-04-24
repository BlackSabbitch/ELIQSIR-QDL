# run.py

import json
import os
import argparse
import torch
from datetime import datetime
from torch_geometric.loader import DataLoader

from extractor import PDBBindOrchestrator
from tokenizer import UniversalPDBBindDataset
from evaluator import Evaluator
from splitter import PDBBindSplitter
from model.model_builder import UHSMBuilder
from parsers.cnn_parser import CNNParser
from parsers.gnn_parser import GNNParser
from logger import log_info, setup_file_logging, get_ascii_plot, log_side_by_side, log_residuals_hist
from model.trainer import HybridTrainer


class ExperimentRunner:
    """
    Orchestrates the full pipeline: data extraction, parsing, splitting,
    training, and evaluation based on a configuration file.
    """

    def __init__(self, config_path: str, extract: bool = False):
        with open(config_path or 'config.json', 'r') as f:
            self.config = json.load(f)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log_info(f"Starting experiment: {self.config['experiment_name']}", stage="EXPERIMENT")
        self.extract = extract

    def prepare(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.config['experiment_name']}_{timestamp}"        
        log_info(f"Experiment signature: {exp_name}", stage="EXPERIMENT")
        self.exp_run_dir = f"runs/{exp_name}"
        self.exp_data_dir = f"datasets/{exp_name}" # Индивидуальная папка для датасетов!
        log_info(f"Datasets folder: {self.exp_data_dir}", stage="EXPERIMENT")
        log_info(f"Run results folder: {self.exp_run_dir}", stage="EXPERIMENT")

        os.makedirs(self.exp_run_dir, exist_ok=True)
        os.makedirs(self.exp_data_dir, exist_ok=True)
        os.makedirs("data/base_datasets", exist_ok=True) # Глобальный кэш

        log_path = os.path.join(self.exp_run_dir, "log.txt")
        setup_file_logging(log_path)
        log_info(f"Log file: {log_path}", stage="EXPERIMENT")

    def run(self):
        trio_str = self.config['model']['graph_encoder']['available']['trio']['protein_ligand_pocket_encoders']
        parsers = []
        for i, char in enumerate(trio_str):
            is_lig = (i == 1)
            if char == 'C': parsers.append(CNNParser(is_ligand=is_lig))
            elif char == 'G': parsers.append(GNNParser(is_ligand=is_lig))
            elif char == 'N': parsers.append(None)
            else: log_info(f"Unknown symbol {char} in the architecture description.", stage="EXPERIMENT")

        orchestrator = PDBBindOrchestrator(parsers, self.config)
        if self.extract: orchestrator.extract_subset("refined")
        df_refined = orchestrator.build_dataset(subset="refined", fmt="pickle", save_dir=self.exp_data_dir)
        df_core = orchestrator.build_dataset(subset="core", fmt="pickle", save_dir=self.exp_data_dir)

        clean_refined = df_refined[~df_refined['pdb_id'].isin(df_core['pdb_id'])]

        train_df, val_df = PDBBindSplitter.split(clean_refined, self.config["splitter"])

        test_df = df_core

        train_path = f"{self.exp_data_dir}/train.pickle"
        val_path   = f"{self.exp_data_dir}/val.pickle"
        test_path  = f"{self.exp_data_dir}/test_core.pickle"

        train_df.to_pickle(train_path)
        test_df.to_pickle(test_path)
        val_df.to_pickle(val_path)

        self.config["dataset"].update({
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
        })

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

        trainer = HybridTrainer(model, evaluator, self.config, self.device)
        best_epoch, _ = trainer.train(train_loader, val_loader, self.exp_run_dir, self.config['training']['save_only_best_epoch'])
        trainer.test(test_loader, self.exp_run_dir, best_epoch)

        log_info("Generating ASCII performance summary...", stage="SUMMARY")

        loss_chart = get_ascii_plot(trainer.history["train_loss"], title="Learning Curve (Loss)")
        log_info(f"Loss Curve:\n" + "\n".join(loss_chart), stage="SUMMARY")

        rmse_chart = get_ascii_plot(trainer.history["val_rmse"], title="Validation RMSE")
        log_info(f"RMSE Curve:\n" + "\n".join(rmse_chart), stage="SUMMARY")

        r_chart = get_ascii_plot(trainer.history["val_pearson"], title="Correlation (Pearson R)")
        log_info(f"Pearson (R) Curve:\n" + "\n".join(r_chart), stage="SUMMARY")

        ci_chart = get_ascii_plot(trainer.history["val_ci"], title="Ranking Accuracy (CI)")
        log_info(f"Concordancy Index Curve:\n" + "\n".join(ci_chart), stage="SUMMARY")

        y_true = trainer.history['best_y_true']
        y_pred = trainer.history['best_y_pred']

        act_vs_pred_chart = get_ascii_plot([y_true, y_pred], title="Predicted = f(Actual)")
        log_info(f"Concordancy Index Curve:\n" + "\n".join(act_vs_pred_chart), stage="SUMMARY")

        log_info("Generating Multi-column ASCII Dashboard...", stage="SUMMARY")

        dashboard_loss_rmse = log_side_by_side(
            trainer.history["train_loss"], "Learning Curve (Loss)",
            trainer.history["val_rmse"], "Validation RMSE"
        )
        log_info(dashboard_loss_rmse, stage="SUMMARY")

        dashboard_r_ci = log_side_by_side(
            trainer.history["val_pearson"], "Correlation (Pearson R)",
            trainer.history["val_ci"], "Ranking Accuracy (CI)"
        )
        log_info(dashboard_r_ci, stage="SUMMARY")

        y_true = trainer.history['best_y_true']
        y_pred = trainer.history['best_y_pred']

        dashboar_act_vs_pred_residuals = log_side_by_side(
            [y_true, y_pred], "Predicted = f(Actual)",
            trainer.history["val_rmse"], "Validation RMSE",
            is_scatter=True
        )
        log_info(dashboar_act_vs_pred_residuals, stage="SUMMARY")

        log_info("Experiment completed successfully.", stage="EXPERIMENT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PDBBind Experiment")

    parser.add_argument('--config', type=str, default='config.json', 
                        help='Path to the configuration JSON file')
    
    # Флаг экстракции (если указан в bash — станет True)
    parser.add_argument('--extract', action='store_true', 
                        help='Extract subset before building dataset')

    args = parser.parse_args()
    runner = ExperimentRunner(config_path=args.config, extract=args.extract)
    runner.prepare()

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