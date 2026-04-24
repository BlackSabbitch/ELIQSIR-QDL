# parsers/__init__.py

from parsers.cnn_parser import CNNParser
from parsers.gnn_parser import GNNParser
from parsers.interaction_graph_parser import InteractionGraphParser


class ParserFactory:
    @classmethod
    def _build_one(cls, mode, is_ligand=False, **kwargs):
        """
        mode: 'C' (CNN), 'G' (GNN), 'E' (EGNN)
        """
        if mode == 'C':
            return CNNParser(is_ligand=is_ligand)
        if mode == 'G':
            return GNNParser(
                is_ligand=is_ligand,
                dist_threshold=kwargs.get('dist_threshold', 10.0),
                ca_only=kwargs.get('ca_only', True))
        if mode in ['E']:
            return InteractionGraphParser(**kwargs)
        raise ValueError(f"Unknown parser mode: {mode}")

    @classmethod
    def build_chain(cls, config_dict):
        """
        На вход: "CGC" и словарь с параметрами из конфига.
        На выход: [prot_parser, lig_parser, pock_parser]
        """
        pairing = {'C': "cnn_params", 'G': "gnn_params", 'E': "egnn_params"}
        mode = config_dict['model']['graph_encoder']['selected']
        available_cfg = config_dict['model']['graph_encoder']['available'][mode]
        config_str = available_cfg['protein_ligand_pocket_encoders']

        parsers = []
        for i, char in enumerate(config_str):
            if char == 'N':
                parsers.append(None)
                continue
            kwargs = available_cfg[pairing[char]]
            p = cls._build_one(
                mode=char, 
                is_ligand=(i == 1),
                **kwargs
            )
            parsers.append(p)
            
        return parsers
