# tokenizer.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PDBBindDataset(Dataset):
    def __init__(self, csv_file, max_prot=1000, max_lig=128, max_pock=63):
        self.df = pd.read_csv(csv_file)
        
        # Конфигурация длин
        self.max_prot = max_prot
        self.max_lig = max_lig
        self.max_pock = max_pock
        
        # Словари — критически важно, чтобы они совпадали с vocab_size в модели
        self.prot_vocab = {c: i+1 for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        self.lig_vocab = {c: i+1 for i, c in enumerate("ABCDEFGHIKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-=[]()#%@+./\\:,;^$")}
        
        # Добавляем символ для неизвестных элементов
        self.prot_vocab['?'] = 0
        self.lig_vocab['?'] = 0

    def encode(self, text, vocab, max_len):
        # Превращаем строку в тензор чисел с дополнением нулями (padding)
        tokens = [vocab.get(c, 0) for c in str(text)[:max_len]]
        padded = tokens + [0] * (max_len - len(tokens))
        return torch.tensor(padded, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # ВНИМАНИЕ: Проверь названия колонок в своем CSV!
        # Я использую те, что были в твоем описании экстрактора
        prot_tensor = self.encode(row['seq'], self.prot_vocab, self.max_prot)
        lig_tensor = self.encode(row['smiles'], self.lig_vocab, self.max_lig)
        
        # Если колонка называется 'pocket_seq' — поправь здесь
        pock_tensor = self.encode(row['pocket_seq'], self.prot_vocab, self.max_pock)
        
        # Таргет (аффинность)
        target = torch.tensor(row['pkd'], dtype=torch.float32)
        
        return prot_tensor, lig_tensor, pock_tensor, target
