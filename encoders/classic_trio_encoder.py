# encoders/classic_trio_encoder.py

import torch
import torch.nn as nn
from encoders.cnn_encoder import FlexCNNBlock
from encoders.gnn_encoder import FlexGNNBlock
from parsers.cnn_parser import CNNParser
from parsers.gnn_parser import GNNParser


class SplitTrioEncoder(nn.Module):
    def __init__(self, prot_enc, lig_enc, pock_enc):
        super().__init__()
        self.prot_enc = prot_enc
        self.lig_enc = lig_enc
        self.pock_enc = pock_enc

        # Сообщаем материнской плате нашу общую размерность
        self.out_dim = self.prot_enc.out_dim + self.lig_enc.out_dim + self.pock_enc.out_dim

        # Считаем общую размерность после конкатенации
        self.out_dim = self.prot_enc.out_dim + self.lig_enc.out_dim + self.pock_enc.out_dim

    def _apply_encoder(self, encoder, data):
        """
        Умный роутер. Определяет природу данных (граф или строка) 
        и вызывает энкодер с правильными аргументами.
        """
        if hasattr(data, 'edge_index'):
            # Это PyG Batch (от GNNParser)
            return encoder(x=data.x, edge_index=data.edge_index, batch=data.batch)
        elif isinstance(data, dict) and 'edge_index' in data:
            # Это сырой словарь графа
            return encoder(x=data['x'], edge_index=data['edge_index'], batch=data['batch'])
        else:
            # Это стандартный тензор (от CNNParser)
            return encoder(data)

    def forward(self, data):
        # Ожидаем кортеж из трех элементов (выдает наш Dataset)
        prot, lig, pock = data
        
        # Микро-уровень: извлекаем фичи (каждый энкодер выдает плоский вектор [Batch, F])
        p_feat = self._apply_encoder(self.prot_enc, prot)
        l_feat = self._apply_encoder(self.lig_enc, lig)
        pk_feat = self._apply_encoder(self.pock_enc, pock)
        
        # Слияние: возвращаем макроскопический вектор комплекса [Batch, sum(F)]
        return torch.cat([p_feat, l_feat, pk_feat], dim=1)


def build_trio_encoder(config_str, config_dict):
    """
    Фабрика, которая собирает нужные энкодеры по строке, например 'CGC'
    C - CNN, G - GNN
    """
    encoders = []
    
    # Конфиг для каждого из 3-х слотов
    for i, char in enumerate(config_str):
        if char == 'C':
            # Достаем нужный словарь в зависимости от позиции (0:prot, 1:lig, 2:pock)
            vocab = config_dict['prot_vocab'] if i in [0, 2] else config_dict['lig_vocab']
            enc = FlexCNNBlock(
                vocab_size=len(vocab), 
                embed_dim=config_dict['embed_dim'], 
                mode=config_dict['cnn_mode']
            )
        elif char == 'G':
            enc = FlexGNNBlock(
                in_channels=3, # или сколько там фичей у твоих атомов
                hidden_channels=config_dict['embed_dim'],
                out_channels=config_dict['embed_dim']
            )
        else:
            raise ValueError(f"Неизвестный тип энкодера: {char}")
            
        encoders.append(enc)
        
    return SplitTrioEncoder(prot_enc=encoders[0], lig_enc=encoders[1], pock_enc=encoders[2])

# Использование:
# trio_encoder = build_trio_encoder("CGC", config)


class TrioPipelineFactory:
    """
    Фабрика для сборки согласованного эксперимента.
    Гарантирует, что Парсеры на CPU и Энкодеры на GPU всегда соответствуют друг другу.
    """
    @staticmethod
    def build(config_str, config_dict):
        """
        config_str: строка вида "CCC", "CGC" и т.д.
        config_dict: словарь с гиперпараметрами (vocab, embed_dim и т.д.)
        Возвращает: (список_парсеров, собранный_SplitTrioEncoder)
        """
        parsers = []
        encoders = []
        
        for i, char in enumerate(config_str):
            # Позиция 1 - это лиганд, 0 и 2 - это белки
            is_ligand = (i == 1)
            vocab = config_dict["dataset"]['lig_vocab'] if is_ligand else config_dict["dataset"]['prot_vocab']
            
            if char == 'C':
                # Создаем согласованную пару: CNN Парсер + CNN Энкодер
                parsers.append(CNNParser(is_ligand=is_ligand))
                
                encoders.append(FlexCNNBlock(
                    vocab_size=len(vocab), 
                    embed_dim=config_dict["model"]['embed_dim'], 
                    mode=config_dict["model"]['cnn_mode']
                ))
                
            elif char == 'G':
                # Создаем согласованную пару: GNN Парсер + GNN Энкодер
                parsers.append(GNNParser(is_ligand=is_ligand))
                
                encoders.append(FlexGNNBlock(
                    in_channels=3, # 3 фичи у белка (XYZ), 3 у лиганда (Num, Deg, Arom)
                    hidden_channels=config_dict['model']['embed_dim'],
                    out_channels=config_dict['model']['embed_dim'],
                    conv_type=config_dict['model']['gnn_mode'],     # <--- ИСПРАВЛЕНИЕ 1
                    heads=config_dict['model'].get('gat_heads', 4), # <--- ИСПРАВЛЕНИЕ 2
                    pool_type=config_dict['model'].get('pooling_type', 'max'),
                    hier_pool_type=config_dict['model'].get('hier_pooling_type', None),
                    hier_pool_ratio=config_dict['model'].get('hier_pooling_ratio', 0.5) # Опционально
                ))
            else:
                raise ValueError(f"Неизвестный тип архитектуры в конфиге: {char}")
                
        # Собираем мета-энкодер
        meta_encoder = SplitTrioEncoder(prot_enc=encoders[0], lig_enc=encoders[1], pock_enc=encoders[2])
        
        return parsers, meta_encoder


"""
from encoders.classic_trio_encoder import TrioPipelineFactory
from model import UniversalHybridSlotModel
from tokenizer import UniversalPDBBindDataset

# 1. Запрашиваем у фабрики согласованный пайплайн (Детектор + АЦП)
# Конфиг-строка может браться из аргументов командной строки (argparse)
parsers, trio_encoder = TrioPipelineFactory.build("CGC", config)

# 2. Инициализируем датасет, отдавая ему готовые парсеры
# В tokenizer.py тебе нужно будет чуть изменить __init__, чтобы он принимал список parsers
train_dataset = UniversalPDBBindDataset(filepath="data/train.csv", parsers=parsers, ...)

# 3. Собираем квантовое ядро
quantum_core = QuantumReUploadingLayer(n_qubits=config['n_qubits'], n_layers=config['q_layers'])

# 4. Собираем финальную материнскую плату
model = UniversalHybridSlotModel(
    graph_encoder=trio_encoder,           # Слот 1 (готов к бою)
    classic_pooler=None,                  # Слот 2A
    quantum_pooler=None,                  # Слот 2B
    global_readout=nn.Identity(),         # Слот 3
    quantum_encoder=quantum_core          # Слот 4
).to(device)
"""