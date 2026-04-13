# encoders/cnn_encoder.py

import torch
import torch.nn as nn

class FlexCNNBlock(nn.Module):
    def __init__(self, vocab_size, embed_dim, mode='parallel', filters=32, kernels=[4, 8, 12]):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if mode == 'parallel':
            # Логика из статьи: независимые фильтры разного размера
            self.convs = nn.ModuleList([
                nn.Conv1d(embed_dim, filters, kernel_size=k, padding=k//2) 
                for k in kernels
            ])
            self.out_dim = filters * len(kernels)
        
        elif mode == 'sequential':
            # Классический глубокий подход: слои друг за другом
            # Каждый следующий слой «видит» комбинации признаков предыдущего
            self.net = nn.Sequential(
                nn.Conv1d(embed_dim, filters, kernel_size=kernels[0], padding=kernels[0]//2),
                nn.ReLU(),
                nn.Conv1d(filters, filters * 2, kernel_size=kernels[1], padding=kernels[1]//2),
                nn.ReLU(),
                nn.Conv1d(filters * 2, filters * 3, kernel_size=kernels[2], padding=kernels[2]//2),
                nn.ReLU()
            )
            self.out_dim = filters * 3 # 32 * 3 = 96
            
        self.relu = nn.ReLU()
        self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        
        if self.mode == 'parallel':
            conv_outputs = [self.relu(conv(x)) for conv in self.convs]
            pooled = [self.global_pool(out).squeeze(-1) for out in conv_outputs]
            return torch.cat(pooled, dim=1)
        
        else: # sequential
            x = self.net(x)
            return self.global_pool(x).squeeze(-1)
