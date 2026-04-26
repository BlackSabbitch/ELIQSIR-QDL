# model/model_builder.py

import torch
import torch.nn as nn
from encoders.trio_encoder import build_trio_encoder
from encoders.QE_encoder import QuantumReUploadingLayer, VQEHead
from typing import Any, Optional, Tuple, Union
from model.model import UniversalHybridSlotModel
from logger import *


class UHSMBuilder:
    @classmethod
    def _get_cfg(cls, entry):
        """Вспомогательный метод для вскрытия структуры selected/available"""
        selected = entry["selected"]
        return selected, entry["available"][selected]

    @staticmethod
    def _initialize_linear_layer(linear: nn.Linear, init_weight, init_bias) -> None:
        """Initialize a linear layer using scalar or list/tensor weights."""
        with torch.no_grad():
            if isinstance(init_weight, list):
                weight_tensor = torch.tensor(init_weight, dtype=linear.weight.dtype, device=linear.weight.device)
                if weight_tensor.numel() == linear.weight.numel():
                    linear.weight.copy_(weight_tensor.view_as(linear.weight))
                elif weight_tensor.numel() == linear.weight.size(1):
                    linear.weight.copy_(weight_tensor.unsqueeze(0))
                else:
                    linear.weight.fill_(float(weight_tensor[0]))
            else:
                linear.weight.fill_(float(init_weight))
            linear.bias.fill_(float(init_bias))

    @classmethod
    def build_component(cls, in_dim, cfg_entry, config: Optional[dict] = None):
        """
        Универсальный конструктор для MLP, Linear и Bottleneck блоков.

        Args:
            in_dim: Входная размерность.
            cfg_entry: Конфигурация выбранного блока.
            config: Полный конфиг для случаев, когда требуется инициализация по общей схеме.
        """
        selected, args = cls._get_cfg(cfg_entry)
        
        # Обрабатываем все типы голов и адаптеров
        if selected in ["MLP", "linear_bottleneck", "linear_tanh"]:
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
            
        elif selected == "Linear":
            return nn.Linear(in_dim, args.get("out_dim", 1))
            
        elif selected == "VQE":
            # Особый случай: Квантовая голова как основная
            q_in_dim = args.get("in_dim", in_dim)
            adapter = nn.Sequential(nn.Linear(in_dim, q_in_dim), nn.Tanh())
            # --- НОВАЯ ИНИЦИАЛИЗАЦИЯ ---
            # Инициализируем веса адаптера маленькими значениями (Xavier Uniform),
            # чтобы углы в начале обучения были распределены около 0.
            # Это гарантирует, что мы не попадем в "мертвые зоны" квантовой схемы сразу.
            nn.init.xavier_uniform_(adapter[0].weight, gain=1.0)
            nn.init.zeros_(adapter[0].bias)
            q_layer = QuantumReUploadingLayer(**args)
            log_debug(f"Q-Layer params: {sum(p.numel() for p in q_layer.parameters())}", stage="DEBUG")
            final_layer = nn.Linear(q_in_dim, 1)

            if config is not None:
                mixer_cfg = config.get("model", {}).get("final_mixer", {})
                selected_mixer = mixer_cfg.get("selected")
                if selected_mixer == "linear_perturbation":
                    mixer_args = mixer_cfg.get("available", {}).get("linear_perturbation", {})
                    init_weight = mixer_args.get("init_weight", 1.0)
                    init_bias = mixer_args.get("init_bias", 0.0)
                    cls._initialize_linear_layer(final_layer, init_weight, init_bias)

            return VQEHead(adapter, q_layer, final_layer)

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
        # total_epochs = config["training"]["epochs"]
        
        m_cfg = config["model"]

        # 1. Encoder (can be complex Trio/Duo logic)
        if m_cfg["graph_encoder"]["selected"] == "trio":
            graph_encoder = build_trio_encoder(config)
        else:
            raise NotImplementedError("Duo mode is not supported yet")
        
        classic_head = cls.build_component(graph_encoder.out_dim, m_cfg["classic_head"], config=config)

        # 3. Quantum branch
        q_encoder = None
        to_quantum_adapter = None
        q_head = None
        
        if m_cfg["quantum_branch"]["enabled"]:
            q_cfg = m_cfg["quantum_branch"]
            to_quantum_adapter = cls.build_component(graph_encoder.out_dim, q_cfg["adapter"], config=config)  # Use MLP/Linear logic
            
            # Initialize our QuantumReUploading class
            _, q_args = cls._get_cfg(q_cfg["encoder"])
            q_encoder = QuantumReUploadingLayer(**q_args)
            
            q_head = cls.build_component(q_encoder.in_dim, q_cfg["head"], config=config)

        # 4. Mixer
        mix_cfg = m_cfg["final_mixer"]
        selected_mix = mix_cfg["selected"]
        if selected_mix == "linear_perturbation":
            args = mix_cfg["available"][selected_mix]
            final_mixer = nn.Linear(args["in_features"], 1)
            # Apply initialization (base + shift)
            cls._initialize_linear_layer(final_mixer, args["init_weight"], args["init_bias"])
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
        log_info("Model built", stage="MODEL")
        
        return model
