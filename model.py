"""
Transformer Decoder-only base model for text generation
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class ModelConfig:
    context_length: int = 256
    vocab_size: int = -1
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = False
    compile: bool = True
    attn_dim: int = n_embd//n_head


# Define feed forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd * 4),
            nn.ReLU(),
            nn.Linear(config.n_embd * 4, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.ffn(x)


# Define Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.Wq = nn.Linear(config.n_embd, config.attn_dim, bias=config.bias)
        self.Wk = nn.Linear(config.n_embd, config.attn_dim, bias=config.bias)
        self.Wv = nn.Linear(config.n_embd, config.attn_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.context_length, config.context_length, requires_grad=False)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        return self.attention(q,k,v,T)
    
    def attention(self,q,k,v,T):
        dk = k.size(-1)
        weights = (q @ k.mT) / math.sqrt(dk)
        weights = weights.masked_fill(self.mask[:T,:T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        return weights@v


# Define Multi-head Attention ｜
class MultiHeadAttention(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config = config
        self.heads = nn.ModuleList([Attention(config) for _ in range(self.config.n_head)])
        self.projection_layer = nn.Linear(self.config.n_embd, self.config.n_embd)
        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection_layer(head_outputs))
        return out


# Define Transformer Block ｜
class TransformerBlock(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mha = MultiHeadAttention(config)
        self.ffn = FeedForwardNetwork(config)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        position = torch.arange(0, config.context_length, requires_grad=False).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        pe = torch.zeros(config.context_length, config.n_embd, requires_grad=False)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:x.size(1),:]


# Define the model ｜
class Model(nn.Module):
    def __init__(self, config:ModelConfig):
        super().__init__()
        self.tok_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embedding = PositionalEncoding(config)
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(config) for _ in range(config.n_layer)] +
                [nn.LayerNorm(config.n_embd)]
        ))
        self.model_out_linear_layer = nn.Linear(config.n_embd, config.vocab_size)
        self.drop = nn.Dropout(config.dropout)
        self.context_length = config.context_length

    def forward(self, idx:torch.Tensor):
        _, T = idx.shape
        pos_emb = self.pos_embedding(idx)
        tok_emb = self.tok_embedding(idx)

        x = self.transformer_blocks(self.drop(tok_emb+pos_emb))
        logits = self.model_out_linear_layer(x)
        return logits

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        # idx is (B,T) array of indices in the current context |
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table |
            idx_crop = idx[:, -self.context_length:]
            # Get predictions |
            logits = self.forward(idx_crop)
            # Get the last time step from logits where the dimensions of the logits are (B,T,C) |
            logits = logits[:, -1, :] / temperature # Divide by temperature |
            # optionally crop the logits to only the top k options |
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to get probabilities |
            probs = F.softmax(input=logits, dim=-1)
            # Sample from the probabilities' distribution. |
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # Append the sampled indexes idx_next to idx |
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
