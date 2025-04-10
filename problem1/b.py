import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class RatingDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = df['userId'].map(user2idx).values
        self.items = df['movieId'].map(item2idx).values
        self.ratings = df['rating'].values.astype(np.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, n_factors=50):
        super().__init__()
        self.user_factors = nn.Embedding(num_users, n_factors)
        self.item_factors = nn.Embedding(num_items, n_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user, item):
        pred = (self.user_factors(user) * self.item_factors(item)).sum(1)
        pred += self.user_bias(user).squeeze() + self.item_bias(item).squeeze()
        return pred
    
    
    

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for user, item, rating in dataloader:
        user, item, rating = user.to(device), item.to(device), rating.to(device)
        optimizer.zero_grad()
        
        preds = model(user, item)
        loss = criterion(preds, rating)
        loss.backward()
        
        optimizer.step()
        total_loss += loss.item() * len(rating)
        
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for user, item, rating in dataloader:
            user, item, rating = user.to(device), item.to(device), rating.to(device)
            preds = model(user, item)
            loss = criterion(preds, rating)
            total_loss += loss.item() * len(rating)
    return total_loss / len(dataloader.dataset)

with open("train.pkl", "rb") as f: 
    train_df = pickle.load(f)
with open("val.pkl", "rb") as f: 
    val_df = pickle.load(f)
#with open("test.pkl", "rb") as f: 
#test_df = pickle.load(f)

print("Validation size:", len(val_df))

unique_users = sorted(set(train_df['userId']) | set(val_df['userId']))
unique_items = sorted(set(train_df['movieId']) | set(val_df['movieId']))
user2idx = {u: i for i, u in enumerate(unique_users)}
item2idx = {m: i for i, m in enumerate(unique_items)}

train_dataset = RatingDataset(train_df, user2idx, item2idx)
val_dataset = RatingDataset(val_df, user2idx, item2idx)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rates = [1e-2, 1e-3]
weight_decays = [1e-1, 1e-2]
results = {}

for lr in learning_rates:
    for wd in weight_decays:
        print(f"\nTraining model with LR={lr}, WD={wd}")
        model = MatrixFactorization(len(user2idx), len(item2idx)).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        
        
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        for epoch in range(10):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        results[(lr, wd)] = {
            'model': model,
            'train_loss': train_losses,
            'val_loss': val_losses,
            'final_val_loss': val_losses[-1]
        }


plt.figure(figsize=(12, 5))

# Training Loss
plt.subplot(1, 2, 1)
for (lr, wd), res in results.items():
    label = f"LR={lr}, WD={wd}"
    plt.plot(res['train_loss'], label=label)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

# Validation Loss
plt.subplot(1, 2, 2)
for (lr, wd), res in results.items():
    label = f"LR={lr}, WD={wd}"
    plt.plot(res['val_loss'], label=label)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

plt.tight_layout()
plt.savefig("mf_loss_curves.png")
plt.show()
