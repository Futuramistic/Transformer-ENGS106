"""
Transformer Decoder-only base model for text generation
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


# Define feed forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.ReLU(),
            nn.Linear(self.d_model * 4, self.d_model),
            nn.Dropout(self.dropout)
        )

    def forward(self, x):
        return self.ffn(x)


# Define Scaled Dot Product Attention
class Attention(nn.Module):
    def __init__(self, d_model, head_size, context_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout
        self.Wq = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wk = nn.Linear(self.d_model, self.head_size, bias=False)
        self.Wv = nn.Linear(self.d_model, self.head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(self.context_length, self.context_length)))
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v

        return output


# Define Multi-head Attention ｜
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout
        self.heads = nn.ModuleList([Attention(self.d_model, self.head_size, self.context_length, self.dropout) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        head_outputs = [head(x) for head in self.heads]
        head_outputs = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.projection_layer(head_outputs))
        return out


# Define Transformer Block ｜
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, head_size, context_length, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, head_size, context_length, dropout)
        self.ffn = FeedForwardNetwork(d_model, dropout)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# Define the model ｜
class Model(nn.Module):
    def __init__(self, h_params):
        super().__init__()
        self.context_length = h_params['context_length']
        self.d_model = h_params['d_model']
        self.learning_rate = h_params['learning_rate']
        self.num_blocks = h_params['num_blocks']
        self.num_heads = h_params['num_heads']
        self.head_size = self.d_model // self.num_heads
        self.dropout = h_params['dropout']
        self.device = h_params['device']
        self.max_token_value = h_params['max_token_value']

        self.token_embedding_lookup_table = nn.Embedding(self.max_token_value, self.d_model)
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(self.d_model, self.num_heads, self.head_size, self.context_length, self.dropout) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        self.model_out_linear_layer = nn.Linear(self.d_model, self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model, device=self.device)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model) | 将position_encoding_lookup_table从(context_length, d_model)更改为(T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(self.device)
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # get the final logits |
        logits = self.model_out_linear_layer(x)

        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C) # reshape logits to (B*T, C) | 将logits重塑为(B*T, C)
            targets_reshaped = targets.view(B * T) # reshape targets to (B*T) | 将targets重塑为(B*T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped) # calculate the loss | 计算损失
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        # idx is (B,T) array of indices in the current context |
        for _ in range(max_new_tokens):
            # Crop idx to the max size of our positional embeddings table |
            idx_crop = idx[:, -self.context_length:]
            # Get predictions |
            logits, loss = self.forward(idx_crop)
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
