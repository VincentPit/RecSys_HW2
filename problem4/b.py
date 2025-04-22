import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, num_users, num_items, hidden_dim=64, max_seq_len=50, num_heads=2, dropout=0.2):
        super(SASRec, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.item_embedding = nn.Embedding(num_items, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Two stacked self-attention blocks
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(2)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Final prediction layer (dot product with item embeddings)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, user_ids, item_seqs):
        """
        user_ids: (batch_size,)
        item_seqs: (batch_size, seq_len)
        """
        seq_len = item_seqs.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=item_seqs.device).unsqueeze(0).expand_as(item_seqs)

        item_emb = self.item_embedding(item_seqs)
        pos_emb = self.position_embedding(position_ids)

        x = item_emb + pos_emb
        x = self.dropout(x)

        # Causal mask: (seq_len, seq_len) with False in the upper triangle
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=item_seqs.device), diagonal=1).bool()

        for block in self.attention_blocks:
            x = block(x, attn_mask)

        x = self.layer_norm(x)  # (batch_size, seq_len, hidden_dim)

        # Predict scores for all items using last hidden state
        last_hidden = x[:, -1, :]  # (batch_size, hidden_dim)
        logits = torch.matmul(last_hidden, self.item_embedding.weight.transpose(0, 1))  # (batch_size, num_items)

        return logits


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate):
        super(SelfAttentionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask):
        # x: (batch_size, seq_len, hidden_dim)
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

if __name__ == "__main__":
    num_users = 1000
    num_items = 5000
    model = SASRec(num_users, num_items)

    user_ids = torch.randint(0, num_users, (4,))
    item_seqs = torch.randint(1, num_items, (4, 50))  # batch of 4, seq_len = 50

    scores = model(user_ids, item_seqs)
    print("Predicted scores:", scores.shape)  # (batch_size, num_items)
