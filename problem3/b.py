import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from problem2.a import NMF
from a import MovieLensTripletDataset


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class BPR(torch.nn.Module):
    def __init__(self, num_users, num_items, rank):
        super().__init__()
        self.nmf = NMF(num_users, num_items, rank)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_ids, pos_items, neg_items):
        pos_scores = self.nmf(user_ids, pos_items)
        neg_scores = self.nmf(user_ids, neg_items)
        return pos_scores, neg_scores

    def bpr_loss(self, pos_scores, neg_scores):
        return -torch.log(self.sigmoid(pos_scores - neg_scores)).mean()

def train_bpr(model, dataloader, val_data, optimizer, epochs=10, k=10, device='cpu'):
    model.to(device)
    recall_scores = []
    bce_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1} start...")
        epoch_start = time.time()

        model.train()
        total_loss = 0.0
        train_start = time.time()

        # tqdm over batches
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch in progress_bar:
            batch_start = time.time()
            users, pos_items, neg_items = [x.to(device) for x in batch]

            # Forward and loss
            forward_start = time.time()
            pos_scores, neg_scores = model(users, pos_items, neg_items)
            loss = model.bpr_loss(pos_scores, neg_scores)
            forward_time = time.time() - forward_start

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            backward_time = time.time() - forward_start - forward_time

            total_loss += loss.item()
            avg_loss_so_far = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss_so_far:.4f}")

            if batch_idx == 0:  # Show timing for the first batch as a sample
                batch_time = time.time() - batch_start
                print(f"  Batch {batch_idx} time: {batch_time:.4f}s")
                print(f"    - Data+to(device): {forward_start - batch_start:.4f}s")
                print(f"    - Forward+loss: {forward_time:.4f}s")
                print(f"    - Backward+step: {backward_time:.4f}s")

        avg_loss = total_loss / len(dataloader)
        bce_losses.append(avg_loss)

        train_time = time.time() - train_start
        print(f"Training time: {train_time:.2f}s")

        eval_start = time.time()
        recall = evaluate_recall(model.nmf, val_data, k=k)
        eval_time = time.time() - eval_start

        recall_scores.append(recall)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Recall@{k}: {recall:.4f}")
        print(f"Evaluation time: {eval_time:.2f}s")
        print(f"Total epoch time: {time.time() - epoch_start:.2f}s")

    return recall_scores, bce_losses

def evaluate_recall(nmf_model, val_data, k=10):
    nmf_model.eval()
    hits = 0
    with torch.no_grad():
        for user, true_item in val_data:
            user_tensor = torch.tensor([user])
            all_items = torch.arange(nmf_model.item_factors.size(0))

            scores = nmf_model(user_tensor.expand_as(all_items), all_items)
            top_k = scores.topk(k).indices
            if true_item in top_k:
                hits += 1
    return hits / len(val_data)



def plot_results(results, ylabel, title, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, values in results.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    filename = title.replace(" ", "_").lower() + ".png"
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close()
    
    
def split_data(interactions, val_ratio=0.1, test_ratio=0.1):
    np.random.seed(1)
    user_pos = {}
    for u, i in interactions:
        user_pos.setdefault(u, set()).add(i)

    train_data, val_data, test_data = [], [], []
    for u in user_pos:
        items = list(user_pos[u])
        np.random.shuffle(items)

        n_total = len(items)
        if n_total < 3:
            continue

        n_val = int(val_ratio * n_total)
        n_test = int(test_ratio * n_total)

        val_items = items[:n_val]
        test_items = items[n_val:n_val+n_test]
        train_items = items[n_val+n_test:]

        for i in train_items:
            train_data.append((u, i))
        for i in val_items:
            val_data.append((u, i))
        for i in test_items:
            test_data.append((u, i))

    return train_data, val_data, test_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ratings_file = "../ml-20m/ratings.csv"

    ratings_df = pd.read_csv(ratings_file)
    interactions = ratings_df[['userId', 'movieId']].drop_duplicates()

    user2id = {uid: i for i, uid in enumerate(interactions['userId'].unique())}
    item2id = {iid: i for i, iid in enumerate(interactions['movieId'].unique())}

    interactions['userId'] = interactions['userId'].map(user2id)
    interactions['movieId'] = interactions['movieId'].map(item2id)

    num_users = len(user2id)
    num_items = len(item2id)

    all_interactions = list(zip(interactions['userId'], interactions['movieId']))
    train_data, val_data, test_data = split_data(all_interactions)

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    neg_k_values = [5, 20]
    learning_rates = [1e-2, 1e-3]
    weight_decays = [1e-1, 1e-2]
    rank = 64
    batch_size = 2048
    epochs = 10

    all_results = {}

    for neg_k in neg_k_values:
        best_recall = 0.0
        best_config = None
        best_recalls = []
        best_losses = []

        for lr in learning_rates:
            for wd in weight_decays:
                print(f"\nTraining BPR with neg_k={neg_k}, lr={lr}, wd={wd}")
                dataset = MovieLensTripletDataset(train_data, num_items, neg_k)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                model = BPR(num_users, num_items, rank).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                recalls, losses = train_bpr(model, dataloader, val_data, optimizer,
                                            epochs=epochs, k=10, device=device)

                if max(recalls) > best_recall:
                    best_recall = max(recalls)
                    best_config = (lr, wd)
                    best_recalls = recalls
                    best_losses = losses
                    best_model_state = model.state_dict()

        label = f"neg_k={neg_k}, lr={best_config[0]}, wd={best_config[1]}"
        all_results[label] = {
            "recall": best_recalls,
            "loss": best_losses
        }
        
        
        model_path = f"checkpoints/bpr_best_model_{label.replace(', ', '_').replace('=', '')}.pth"
        torch.save(best_model_state, model_path)
        print(f"Saved best model for {label} to {model_path}")

    #Plot
    plot_results({k: v["recall"] for k, v in all_results.items()}, ylabel="Recall@10", title="Recall@10 vs Epochs")
    plot_results({k: v["loss"] for k, v in all_results.items()}, ylabel="BPR Loss", title="BPR Loss vs Epochs")

    #Evaluation
    print("\nFinal Test Set Evaluation:")
    for label, result in all_results.items():
        neg_k = int(label.split(',')[0].split('=')[1])
        print(f"Using best model from {label}")
        dataset = MovieLensTripletDataset(train_data, num_items, neg_k)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = BPR(num_users, num_items, rank).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(label.split(',')[1].split('=')[1]),
                                     weight_decay=float(label.split(',')[2].split('=')[1]))

        train_bpr(model, dataloader, val_data, optimizer, epochs=epochs, k=10, device=device)
        recall_test = evaluate_recall(model.nmf, test_data, k=10)
        print(f"Recall@10 on Test Set ({label}): {recall_test:.4f}")

if __name__ == "__main__":
    main()