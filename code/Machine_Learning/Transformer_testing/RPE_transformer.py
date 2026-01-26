import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

def relative_shift(x):
    """
    x: (B, H, L, 2L-1)
    returns: (B, H, L, L) aligned relative attention
    """
    B, H, L, _ = x.size()

    # Step 1: pad on the left
    zero_pad = torch.zeros((B, H, L, 1), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)  # (B, H, L, 2L)

    # Step 2: reshape
    x_padded = x_padded.view(B, H, -1, L)  # (B, H, 2L, L)

    # Step 3: slice out the correct part
    x = x_padded[:, :, 1:, :]              # (B, H, 2L-1, L)

    # Step 4: final reshape and crop
    x = x.view(B, H, L, -1)                 # (B, H, L, 2L-1)
    x = x[:, :, :, :L]                      # (B, H, L, L)

    return x



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Relative positional embeddings (2L-1 possible distances)
        self.rel_emb = nn.Embedding(2 * max_len, self.d_k)

        # Global content and position bias (u and v in paper)
        self.u = nn.Parameter(torch.zeros(num_heads, self.d_k))
        self.v = nn.Parameter(torch.zeros(num_heads, self.d_k))

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.size()

        # Project and split heads
        q = self.split_heads(self.W_q(Q))  # (B, H, L, D)
        k = self.split_heads(self.W_k(K))
        v = self.split_heads(self.W_v(V))

        # ---- Content-based term (Q + u) K^T ----
        AC = torch.matmul(q + self.u.unsqueeze(0).unsqueeze(2),
                          k.transpose(-2, -1))  # (B, H, L, L)

        # ---- Position-based term (Q + v) R^T ----
        # Relative positions: (L-1 ... -L+1)
        pos = torch.arange(L - 1, -L, -1, device=Q.device)
        pos = pos + self.max_len  # shift to positive indices

        r = self.rel_emb(pos)  # (2L-1, D)

        # (B, H, L, 2L-1)
        BD = torch.matmul(q + self.v.unsqueeze(0).unsqueeze(2),
                          r.transpose(0, 1))

        # Align relative positions
        BD = relative_shift(BD)
        BD = BD[:, :, :, :L]

        # Combine and scale
        attn_score = (AC + BD) / math.sqrt(self.d_k)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float('-inf'))

        attn_probs = torch.softmax(attn_score, dim=-1)

        output = torch.matmul(attn_probs, v)  # (B, H, L, D)
        output = self.W_o(self.combine_heads(output))

        return output

    

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, max_len):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, max_len)
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

        self.encoding_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, max_len=len_spectra)
            for _ in range(num_layers)
        ])

        # 🔹 Attention pooling instead of mean pooling
        self.pooling = AttentionPooling(d_model)

        self.fc = nn.Linear(d_model, len_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Embed spectra (no positional encoding)
        x = self.dropout(self.encoder_embedding(src))

        # Transformer encoder
        for enc_layer in self.encoding_layers:
            x = enc_layer(x)

        # 🔹 Attention pooling over frequency axis
        pooled = self.pooling(x)   # (B, D)

        output = self.fc(pooled)
        return output
    
    def forward_with_pooling_weights(self, src):
        x = self.dropout(self.encoder_embedding(src))

        for enc_layer in self.encoding_layers:
            x = enc_layer(x)

        pooled, weights = self.pooling(x, return_weights=True)
        output = self.fc(pooled)

        return output, weights


if __name__ == "__main__":

    batch = 10
    L = 100
    d_model = 126
    num_heads = 6

    input = torch.randn(batch, L, d_model)

    mha = MultiHeadAttention(d_model, num_heads, max_len=L)

    out = mha.forward(input, input, input)
    print(out.shape)   # (10, 100, 126)
