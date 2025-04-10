import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# GMF
class GMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

    def forward(self, user_indices, item_indices):
        #print(f"Max user index: {user_indices.max().item()}, Expected max: {self.user_embedding.num_embeddings - 1}")
        #print(f"Min user index: {user_indices.min().item()}, Expected min: 0")
        user_vecs = self.user_embedding(user_indices)
        #print(f"Max item index: {item_indices.max().item()}, Expected max: {self.item_embedding.num_embeddings - 1}")
        #print(f"Min item index: {item_indices.min().item()}, Expected min: 0")
        #print(f"Shape of item_indices: {item_indices.shape}")
        item_vecs = self.item_embedding(item_indices)
        return torch.mul(user_vecs, item_vecs)  # element-wise product


# MLP
class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(MLP, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.fc1 = nn.Linear(latent_dim * 2, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, user_indices, item_indices):
        user_vecs = self.user_embedding(user_indices)
        item_vecs = self.item_embedding(item_indices)
        x = torch.cat([user_vecs, item_vecs], dim=-1)
        return self.relu(self.fc1(x))


#NMF= GMF + MLP
class NMF(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(NMF, self).__init__()
        self.gmf = GMF(num_users, num_items, latent_dim)
        self.mlp = MLP(num_users, num_items, latent_dim)
        self.predict_layer = nn.Linear(latent_dim * 2, 1)

    def forward(self, user_indices, item_indices):
        gmf_out = self.gmf(user_indices, item_indices)  # [batch, latent_dim]
        mlp_out = self.mlp(user_indices, item_indices)  # [batch, latent_dim]
        concat = torch.cat([gmf_out, mlp_out], dim=1)  # [batch, latent_dim * 2]
        return self.predict_layer(concat).squeeze()


if __name__ == "__main__":
    # Sample dimensions
    num_users = 138493
    num_items = 26673
    latent_dim = 50

    # Create model instances
    gmf_model = GMF(num_users, num_items, latent_dim)
    mlp_model = MLP(num_users, num_items, latent_dim)
    nmf_model = NMF(num_users, num_items, latent_dim)

    # Print summaries
    print("GMF Model Summary:")
    summary(gmf_model, input_data=(torch.randint(0, num_users, (64,)), torch.randint(0, num_items, (64,))))

    print("\nMLP Model Summary:")
    summary(mlp_model, input_data=(torch.randint(0, num_users, (64,)), torch.randint(0, num_items, (64,))))

    print("\nNMF Model Summary:")
    summary(nmf_model, input_data=(torch.randint(0, num_users, (64,)), torch.randint(0, num_items, (64,))))
