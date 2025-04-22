import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from problem2.a import NMF
from a import MovieLensTripletDataset
from b import BPR, evaluate_recall

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class DPR(nn.Module):
    def __init__(self, num_users, num_items, rank):
        super(DPR, self).__init__()
        self.user_embedding = nn.Embedding(num_users, rank)
        self.item_embedding = nn.Embedding(num_items, rank)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        # Parameters for Gaussian mean and log-variance
        self.fc_mean = nn.Linear(rank, 1)
        self.fc_logvar = nn.Linear(rank, 1)

        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, users, items):
        u_emb = self.user_embedding(users)
        i_emb = self.item_embedding(items)
        interaction = u_emb * i_emb

        mean = self.fc_mean(interaction).squeeze()
        logvar = self.fc_logvar(interaction).squeeze()
        std = torch.exp(0.5 * logvar)

        return mean, std, logvar

def rsp_loss(mean_pos, std_pos, mean_neg, std_neg):
    diff = (mean_pos - mean_neg) / (std_pos + 1e-8)
    return -torch.log(torch.sigmoid(diff)).mean()

def kl_regularization(mean, logvar):
    return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())


def train_dpr(model, dataloader, val_data, optimizer, num_items, epochs, k, device):
    model.train()
    recalls = []
    losses = []

    for epoch in range(epochs):
        total_loss = 0
        for user, pos_item, neg_item in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            user = user.to(device)
            pos_item = pos_item.to(device)
            neg_item = neg_item.to(device)

            mean_pos, std_pos, logvar_pos = model(user, pos_item)
            mean_neg, std_neg, logvar_neg = model(user, neg_item)

            loss_rsp = rsp_loss(mean_pos, std_pos, mean_neg, std_neg)
            loss_kl = kl_regularization(mean_pos, logvar_pos) + kl_regularization(mean_neg, logvar_neg)
            loss = loss_rsp + 1e-3 * loss_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        recall = evaluate_recall(model,num_items,val_data, k, device, is_dpr=True)
        losses.append(total_loss / len(dataloader))
        recalls.append(recall)
        print(f"Epoch {epoch+1}: Loss={losses[-1]:.4f}, Recall@{k}={recall:.4f}")

    return recalls, losses


def plot_score_cdfs(model_bpr, model_dpr, num_users, num_items, device, save_path="score_cdfs.png"):
    users = torch.randint(0, num_users, (1000,), device=device)
    items = torch.arange(num_items, device=device)

    bpr_scores = []
    dpr_means = []

    for user in tqdm(users, desc="Computing scores for CDF"):
        user_tensor = user.repeat(num_items)
        items_tensor = items

        with torch.no_grad():
            bpr_score = model_bpr(user_tensor, items_tensor)
            mean, _, _ = model_dpr(user_tensor, items_tensor)

        bpr_scores.append(bpr_score.cpu().numpy())
        dpr_means.append(mean.cpu().numpy())

    bpr_scores = np.array(bpr_scores).flatten()
    dpr_means = np.array(dpr_means).flatten()

    plt.figure(figsize=(8, 5))
    for scores, label in zip([bpr_scores, dpr_means], ["BPR", "DPR"]):
        sorted_scores = np.sort(scores)
        cdf = np.arange(len(sorted_scores)) / float(len(sorted_scores))
        plt.plot(sorted_scores, cdf, label=label)

    plt.title("CDF of Score Distributions")
    plt.xlabel("Score")
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
def load_implicit_feedback_data(ratings_path="../ml-20m/ratings.csv", neg_k=2, batch_size=256, val_ratio=0.1):

    # Load the ratings file
    ratings_df = pd.read_csv(ratings_path)
    print("Ratings loaded.")
    interactions = ratings_df[['userId', 'movieId']].drop_duplicates()

    # Map userId and movieId to indices starting from 0
    user2id = {uid: i for i, uid in enumerate(interactions['userId'].unique())}
    item2id = {iid: i for i, iid in enumerate(interactions['movieId'].unique())}

    interactions['userId'] = interactions['userId'].map(user2id)
    interactions['movieId'] = interactions['movieId'].map(item2id)

    num_users = len(user2id)
    num_items = len(item2id)
    interaction_tuples = list(zip(interactions['userId'], interactions['movieId']))
    train_interactions, val_interactions = train_test_split(interaction_tuples, test_size=val_ratio, random_state=42)
    train_dataset = MovieLensTripletDataset(interactions=train_interactions, num_items=num_items, neg_k=neg_k)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return num_users, num_items, train_loader, val_interactions



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_users, num_items, train_loader, val_data = load_implicit_feedback_data()

    rank = 64
    epochs = 10
    k = 10
    lr = 1e-3

    # Train BPR for comparison (assume pretrained or load if exists)
    model_bpr = BPR(num_users, num_items, rank).to(device)
    bpr_path = "checkpoints/bpr_models/bpr_best_model_neg_k5_lr0.01_wd0.1.pth"
    if os.path.exists(bpr_path):
        model_bpr.load_state_dict(torch.load(bpr_path))
    else:
        raise FileNotFoundError(f"BPR model not found at {bpr_path}")
    model_bpr.eval()

    # Train DPR
    model_dpr = DPR(num_users, num_items, rank).to(device)
    optimizer = optim.Adam(model_dpr.parameters(), lr=lr)
    recall_dpr, losses_dpr = train_dpr(model_dpr, train_loader, val_data, optimizer, num_items, epochs, k, device)
    torch.save(model_dpr.state_dict(), "dpr_model.pth")
    full_recall = evaluate_recall(model_dpr,num_items, val_data, k, device, full=True, is_dpr=True)
    print(f"DPR Full Recall@{k}: {full_recall:.4f}")

    # Save results to file
    with open("results.txt", "a") as f:
        f.write(f"DPR Recall@{k} (sampled): {recall_dpr[-1]:.4f}\n")
        f.write(f"DPR Recall@{k} (full): {full_recall:.4f}\n")

    plot_score_cdfs(model_bpr, model_dpr, num_users, num_items, device, save_path="score_cdfs.png")


if __name__ == "__main__":
    main()