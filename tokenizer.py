# tokenizer.py

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class UniversalPDBBindDataset(Dataset):
    def __init__(self, filepath, max_prot=1000, max_lig=128, max_pock=63):
        # 1. Автоматическое определение формата файла
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
        elif filepath.endswith(('.pkl', '.pickle')):
            self.df = pd.read_pickle(filepath)
        elif filepath.endswith('.parquet'):
            self.df = pd.read_parquet(filepath)
        else:
            raise ValueError("Неподдерживаемый формат файла. Используйте .csv, .pkl или .parquet")
        
        self.max_prot = max_prot
        self.max_lig = max_lig
        self.max_pock = max_pock
        
        # Словари для CNN-токенизации (оставляем без изменений)
        self.prot_vocab = {c: i+1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.lig_vocab = {c: i+1 for i, c in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-=[]()#%@+./\\:,;^$")}
        self.prot_vocab['?'] = 0
        self.lig_vocab['?'] = 0

    def __len__(self):
        return len(self.df)

    def _encode_sequence(self, text, vocab, max_len):
        """Обработчик строк (для CNN)"""
        tokens = [vocab.get(c, 0) for c in str(text)[:max_len]]
        padded = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)

    def _encode_graph(self, graph_dict):
        """Обработчик словарей (для GNN)"""
        if graph_dict is None or not isinstance(graph_dict, dict):
            # Fallback на случай битых данных
            return Data(x=torch.zeros((1, 3)), edge_index=torch.empty((2, 0), dtype=torch.long))
            
        x = torch.tensor(graph_dict['x'], dtype=torch.float32)
        edge_index = torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        
        # PyG требует форму [2, num_edges]. Если пришло [num_edges, 2] - транспонируем
        if edge_index.numel() > 0 and edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
            
        return Data(x=x, edge_index=edge_index)

    def _process_item(self, item, vocab=None, max_len=None):
        """Умный роутер: определяет тип данных и вызывает нужный энкодер"""
        if isinstance(item, dict):
            return self._encode_graph(item)
        else:
            return self._encode_sequence(item, vocab, max_len)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Токенизатор сам разберется, что лежит внутри (строка или граф)
        prot_data = self._process_item(row['seq'], self.prot_vocab, self.max_prot)
        lig_data  = self._process_item(row['smiles'], self.lig_vocab, self.max_lig)
        pock_data = self._process_item(row['pocket_seq'], self.prot_vocab, self.max_pock)
        
        # Таргет (аффинность)
        target = torch.tensor(row['pkd'], dtype=torch.float32)
        
        return prot_data, lig_data, pock_data, target
