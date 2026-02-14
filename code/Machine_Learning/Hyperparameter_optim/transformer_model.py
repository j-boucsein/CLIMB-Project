import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model  # Model's dimension
        self.num_heads = num_heads  # Number of attention heads
        self.d_k = d_model // num_heads  # Dimension of each head's key, querry, and value

        # Linear layers for transoforming inputs
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calvulate attention scores
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_score, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output 
    
    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(2, 1)
    
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # Perfomr scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))

        return output
    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class SpectraTokenEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, 468)
        return: (batch_size, 468, embed_dim)
        """
        x = x.unsqueeze(-1)        # (B, 468, 1)
        return self.proj(x)        # (B, 468, embed_dim)
    

class AttentionPooling(nn.Module):
    """
    Learns a weighted average over the sequence dimension.
    """

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, x, return_weights=False):
        # x: (B, L, D)

        # (B, L)
        scores = torch.matmul(x, self.query) / self.scale
        attn_weights = torch.softmax(scores, dim=1)

        # (B, D)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)

        if return_weights:
            return pooled, attn_weights
        else:
            return pooled
    

class Transformer(nn.Module):
    def __init__(self, len_spectra, len_output, d_model, num_heads, num_layers, d_ff, dropout):
        super().__init__()
        self.encoder_embedding = SpectraTokenEmbedding(d_model)
        self.positional_encoding = PositionalEncoding(d_model, len_spectra)

        self.encoding_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.pooling = AttentionPooling(d_model)

        self.fc = nn.Linear(d_model, len_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        input_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = input_embedded
        for enc_layer in self.encoding_layers:
            enc_output = enc_layer(enc_output)

        pooled = self.pooling(enc_output)

        output = self.fc(pooled)
        return output


if __name__ == "__main__":


    batch = 10
    L = 100
    d_model=126
    num_heads = 6
    input = torch.randn(batch, L, d_model)

    mha = MultiHeadAttention(d_model, num_heads)

    out = mha.forward(input, input, input)
    print(out.shape)