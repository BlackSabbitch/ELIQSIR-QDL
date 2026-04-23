# model/model_builder.py

import torch
import torch.nn as nn
from encoders.trio_encoder import build_trio_encoder
from encoders.original_quantum_encoder import QuantumReUploadingLayer
from typing import Optional, Tuple, Union
from model.model import UniversalHybridSlotModel
from logger import *


class UHSMBuilder:
    @classmethod
    def _get_cfg(cls, entry):
        """Вспомогательный метод для вскрытия структуры selected/available"""
        selected = entry["selected"]
        return selected, entry["available"][selected]

    @classmethod
    def build_component(cls, in_dim, cfg_entry):
        """
        Универсальный конструктор для MLP, Linear и Bottleneck блоков.
        """
        selected, args = cls._get_cfg(cfg_entry)
        
        # Обрабатываем все типы голов и адаптеров
        if selected in ["mlp_head", "linear_bottleneck", "linear_tanh"]:
            layers = []
            curr_dim = in_dim
            # Поддерживаем и hidden_dims, и hidden_layers для гибкости
            h_dims = args.get("hidden_dims", args.get("hidden_layers", []))
            
            for h in h_dims:
                layers.append(nn.Linear(curr_dim, h))
                layers.append(getattr(nn, args.get("activation", "ReLU"))())
                if args.get("dropout", 0) > 0:
                    layers.append(nn.Dropout(args["dropout"]))
                curr_dim = h
            
            layers.append(nn.Linear(curr_dim, args.get("out_dim", 1)))
            return nn.Sequential(*layers)
            
        elif selected == "linear_head":
            return nn.Linear(in_dim, args.get("out_dim", 1))
            
        elif selected == "quantum_head":
            # Особый случай: Квантовая голова как основная
            n_qubits = args.get("n_qubits", 4)
            adapter = nn.Sequential(nn.Linear(in_dim, n_qubits), nn.Tanh())
            q_layer = QuantumReUploadingLayer(**args)
            return nn.Sequential(adapter, q_layer, nn.Linear(n_qubits, 1))

        raise ValueError(f"Unknown component type: {selected}")

    @classmethod
    def build_model_from_config(cls, config: dict) -> UniversalHybridSlotModel:
        """
        Build complete model from configuration dictionary.

        Constructs all necessary components (encoder, heads, adapters, mixers)
        based on the provided configuration and assembles them into a
        UniversalHybridSlotModel instance.

        Args:
            config: Complete configuration dictionary containing model specifications.

        Returns:
            Fully configured UniversalHybridSlotModel instance.

        Example:
            >>> with open('config.json', 'r') as f:
            ...     config = json.load(f)
            >>> model = UHSMBuilder.build_model_from_config(config)
        """
        m_cfg = config["model"]
        
        # 1. Encoder (can be complex Trio/Duo logic)
        if m_cfg["graph_encoder"]["selected"] == "trio":
            graph_encoder = build_trio_encoder(config)
        else:
            raise NotImplementedError("Duo mode is not supported yet")
        
        classic_head = cls.build_component(graph_encoder.out_dim, m_cfg["classic_head"])

        # 3. Quantum branch
        q_encoder = None
        to_quantum_adapter = None
        q_head = None
        
        if m_cfg["quantum_branch"]["enabled"]:
            q_cfg = m_cfg["quantum_branch"]
            to_quantum_adapter = cls.build_component(graph_encoder.out_dim, q_cfg["adapter"])  # Use MLP/Linear logic
            
            # Initialize our QuantumReUploading class
            _, q_args = cls._get_cfg(q_cfg["encoder"])
            q_encoder = QuantumReUploadingLayer(**q_args)
            
            q_head = cls.build_component(q_encoder.n_qubits, q_cfg["head"])

        # 4. Mixer
        mix_cfg = m_cfg["final_mixer"]
        selected_mix = mix_cfg["selected"]
        if selected_mix == "linear_perturbation":
            args = mix_cfg["available"][selected_mix]
            final_mixer = nn.Linear(args["in_features"], 1)
            # Apply initialization (base + shift)
            with torch.no_grad():
                final_mixer.weight.fill_(1.0)  # or from config: args["init_weight"]
                final_mixer.bias.fill_(args["init_bias"])
        else:
            final_mixer = None

        # Final assembly
        model = UniversalHybridSlotModel(
            graph_encoder=graph_encoder,
            classic_head=classic_head,
            to_quantum_adapter=to_quantum_adapter,
            quantum_encoder=q_encoder,
            quantum_head=q_head,
            final_mixer=final_mixer
        )
        
        return model
