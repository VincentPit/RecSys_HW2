import torch
import pandas as pd
import random
from torch import nn

# Dummy model
class DummyModel(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, 8)
        self.item_embedding = nn.Embedding(num_items, 8)

    def forward(self, users, items):
        u = self.user_embedding(users)
        i = self.item_embedding(items)
        return (u * i).sum(dim=1)  # Dot product

# Dummy evaluation function (copied with fix)
@torch.no_grad()
def sampled_evaluate(model, val_df, num_items, device, K=10):
    model.eval()
    user_item_dict = val_df.groupby('userId')['movieId'].apply(set).to_dict()
    recalls = []

    for user in user_item_dict:
        positives = list(user_item_dict[user])
        candidate_items = set(range(num_items)) - set(positives)
        if len(candidate_items) < 99:
            continue  # Skip if not enough negatives to sample from

        negatives = random.sample(list(candidate_items), 99)
        items = torch.tensor([positives[0]] + negatives).long().to(device)
        users = torch.tensor([user] * 100).long().to(device)

        scores = model(users, items)
        _, indices = torch.topk(scores, K)
        top_k_items = items[indices].tolist()

        recalls.append(1 if positives[0] in top_k_items else 0)

    return sum(recalls) / len(recalls) if recalls else 0.0

# Construct dummy val_df
val_data = {
    'userId': [0, 0, 1, 2, 2, 2],
    'movieId': [1, 3, 5, 2, 4, 6]
}
val_df = pd.DataFrame(val_data)

# Setup
num_users = val_df['userId'].max() + 1
num_items = 10  # More than any movieId in val_df
device = 'cpu'

# Instantiate and test
model = DummyModel(num_users, num_items).to(device)
recall = sampled_evaluate(model, val_df, num_items, device)

print(f"Sampled Recall@10: {recall:.4f}")
