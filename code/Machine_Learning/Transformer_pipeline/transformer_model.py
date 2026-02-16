import torch
import torch.nn as nn
import math


class SpectraTokenEmbedding(nn.Module):
    """
    Linear Embedding for the input spectra
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)

    def forward(self, x):
        """
        x: (batch_size, len_spectra)
        return: (batch_size, len_spectra, d_model)
        """
        x = x.unsqueeze(-1)
        return self.proj(x)


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for the input spectra
    """
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        x: (batch_size, len_spectra, d_model)
        return: (batch_size, len_spectra, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention inside Transformer block
    """
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q: (batch_size, len_spectra, num_heads, d_k)
        K: (batch_size, len_spectra, num_heads, d_k)
        V: (batch_size, len_spectra, num_heads, d_k)
        return: (batch_size, len_spectra, num_heads, d_k)
        """
        # Calculate attention scores
        attn_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_score, dim=-1)
        
        # Multiply by values to obtain the final output
        output = torch.matmul(attn_probs, V)
        return output 
    
    def split_heads(self, x):
        """
        x: (batch_size, len_spectra, d_model)
        return: (batch_size, len_spectra, num_heads, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(2, 1)
    
    def combine_heads(self, x):
        """
        x: (batch_size, num_heads, len_spectra, d_k)
        return: (batch_size, len_spectra, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Q: (batch_size, len_spectra, d_model)
        K: (batch_size, len_spectra, d_model)
        V: (batch_size, len_spectra, d_model)
        return: (batch_size, len_spectra, d_model)
        """

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))

        return output
    

class PositionWiseFeedForward(nn.Module):
    """
    Feed Forward Layer inside Transformer block
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (batch_size, len_spectra, d_model)
        return: (batch_size, len_spectra, d_model)
        """
        return self.fc2(self.relu(self.fc1(x)))
    

class EncoderLayer(nn.Module):
    """
    Implementation of one Transformer Layer
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (batch_size, len_spectra, d_model)
        return: (batch_size, len_spectra, d_model)
        """
        attn_output = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class AttentionPooling(nn.Module):
    """
    Learns a weighted average over the sequence dimension.
    """

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        """
        x: (batch_size, len_spectra, d_model)
        return: (batch_size, d_model)
        """

        scores = torch.matmul(x, self.query) / self.scale
        attn_weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        
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
        """
        src: (batch_size, len_spectra)
        return: (batch_size, len_output)
        """
        input_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))

        enc_output = input_embedded
        for enc_layer in self.encoding_layers:
            enc_output = enc_layer(enc_output)

        pooled = self.pooling(enc_output)

        output = self.fc(pooled)
        return output


if __name__ == "__main__":
    ...