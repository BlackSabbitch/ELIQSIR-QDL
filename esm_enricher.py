# esm_enricher.py

import torch
import esm


class ESMEnricher:
    def __init__(self, model_name="esm2_t33_650M_UR50D", device="cuda"):
        print(f"Loading ESM model {model_name}...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def get_sequence_embedding(self, sequence):
        """Возвращает векторное представление всей последовательности (для CNN-ветки)"""
        # Подготавливаем данные
        data = [("seq", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Прогон через ESM
        results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]

        # Усредняем эмбеддинги всех аминокислот, чтобы получить вектор белка (размер 1280)
        sequence_representation = token_representations[0, 1 : -1].mean(0)
        return sequence_representation.cpu().numpy()