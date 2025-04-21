from a import NMF
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
from tqdm import tqdm 
import time

class MovieLensImplicitDataset(Dataset):
    def __init__(self, interactions_df, num_items, num_negatives=4, device = "cpu"):
        self.users = []
        self.items = []
        self.labels = []
        self.num_items = num_items

        user_item_set = set(zip(interactions_df['userId'], interactions_df['movieId']))

        for user, item in user_item_set:
            # Positive 
            self.users.append(user)
            self.items.append(item)
            self.labels.append(1)

            # Negative 
            for _ in range(num_negatives):
                neg_item = random.randint(0, num_items - 1)  # Ensure it falls within the reindexed item range
                while (user, neg_item) in user_item_set:  # Ensure it's a negative sample
                    neg_item = random.randint(0, num_items - 1)
                self.users.append(user)
                self.items.append(neg_item)
                self.labels.append(0)


    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        
        user, item, label = (torch.LongTensor([self.users[idx]]).squeeze(),
            torch.LongTensor([self.items[idx]]).squeeze(),
            torch.FloatTensor([self.labels[idx]]).squeeze())
        user, item, label = user.to(device), item.to(device), label.to(device)
        
        return (
            user, item, label
        )

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    train_start = time.time()

    data_load_time = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        batch_start = time.time()

        # Time data loading separately
        data_load_start = time.time()
        user, item, label = batch
        data_load_time_batch = time.time() - data_load_start
        data_load_time += data_load_time_batch

        # Forward pass and loss
        forward_start = time.time()
        preds = model(user, item)
        loss = criterion(preds, label)
        forward_time = time.time() - forward_start

        # Backward pass and optimization
        backward_start = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        backward_time = time.time() - backward_start

        total_loss += loss.item()
        batch_time = time.time() - batch_start

        if batch_idx == 0:  # Print timing only for the first batch
            print(f"\nBatch {batch_idx} time: {batch_time:.4f}s")
            print(f"  - data loading + to(device): {data_load_time_batch:.4f}s")
            print(f"  - forward+loss: {forward_time:.4f}s")
            print(f"  - backward+step: {backward_time:.4f}s")

    total_time = time.time() - train_start
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Total DataLoader time: {data_load_time:.2f}s ({100 * data_load_time / total_time:.2f}%)")

    return total_loss / len(dataloader)
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

    return sum(recalls) / len(recalls)

@torch.no_grad()
def full_evaluate(model, test_df, all_user_item_df, num_items, device, K_list=[10, 50]):
    model.eval()
    user_item_dict = test_df.groupby('userId')['movieId'].apply(set).to_dict()
    train_item_dict = all_user_item_df.groupby('userId')['movieId'].apply(set).to_dict()

    results = {k: [] for k in K_list}
    ndcgs = {k: [] for k in K_list}

    for user in user_item_dict:
        test_items = list(user_item_dict[user])
        seen_items = train_item_dict.get(user, set())
        candidates = list(set(range(1, num_items)) - seen_items)

        users = torch.tensor([user] * len(candidates)).long().to(device)
        items = torch.tensor(candidates).long().to(device)
        scores = model(users, items)

        _, indices = torch.topk(scores, max(K_list))
        top_k_items = items[indices].tolist()

        for K in K_list:
            top_items = set(top_k_items[:K])
            hits = sum([1 for item in test_items if item in top_items])
            dcg = sum([1 / torch.log2(torch.tensor(i + 2.).float()) for i, item in enumerate(top_k_items[:K]) if item in test_items])
            idcg = sum([1 / torch.log2(torch.tensor(i + 2.).float()) for i in range(min(len(test_items), K))])

            results[K].append(hits / len(test_items))
            ndcgs[K].append((dcg / idcg).item() if idcg > 0 else 0)

    return {
        'Recall@10': sum(results[10]) / len(results[10]),
        'Recall@50': sum(results[50]) / len(results[50]),
        'NDCG@10': sum(ndcgs[10]) / len(ndcgs[10]),
        'NDCG@50': sum(ndcgs[50]) / len(ndcgs[50]),
    }

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open("../problem1/train.pkl", "rb") as f: 
        train_df = pickle.load(f)
    with open("../problem1/val.pkl", "rb") as f: 
        val_df = pickle.load(f)
    
    train_df = train_df[train_df['rating'] > 0]
    val_df = val_df[val_df['rating'] > 0]
    
    unique_user_ids = train_df['userId'].unique()
    unique_item_ids = train_df['movieId'].unique()

    # Mapping original IDs to contiguous indices
    user2id = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    item2id = {iid: idx for idx, iid in enumerate(unique_item_ids)}
    
    num_users = len(user2id)+1
    num_items = len(item2id)+1
    
    # Reindexing 
    train_df['userId'] = train_df['userId'].map(user2id)
    train_df['movieId'] = train_df['movieId'].map(item2id)
    val_df = val_df[val_df['userId'].isin(user2id)]  
    val_df = val_df[val_df['movieId'].isin(item2id)]  

    val_df['userId'] = val_df['userId'].map(user2id)
    val_df['movieId'] = val_df['movieId'].map(item2id)

    # Ensure no missing mappings
    assert not train_df['userId'].isnull().any(), "Missing userId in mapping!"
    assert not train_df['movieId'].isnull().any(), "Missing movieId in mapping!"
    assert not val_df['userId'].isnull().any(), "Missing userId in mapping!"
    assert not val_df['movieId'].isnull().any(), "Missing movieId in mapping!"

    # Prepare the datasets and dataloaders
    train_dataset = MovieLensImplicitDataset(train_df, num_items, num_negatives=4)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # Hyperparameters
    lrs = [1e-2, 1e-3]
    wds = [1e-1, 1e-2]

    results = {}

    for lr in lrs:
        for wd in wds:
            print(f"Training with LR={lr}, WD={wd}")
            model = NMF(num_users, num_items, latent_dim=50).to(device)
            #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            
            #optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr, weight_decay=wd)
            
            criterion = nn.BCEWithLogitsLoss()

            train_losses = []
            val_recalls = []
            
            

            for epoch in tqdm(range(10), desc="Epochs", ncols=100):
                train_loss = train(model, train_loader, optimizer, criterion, device)
                train_losses.append(train_loss)
                
                # Evaluation
                recall = sampled_evaluate(model, val_df, num_items, device)
                val_recalls.append(recall)

                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Recall@10 = {recall:.4f}")

            results[(lr, wd)] = {'train_loss': train_losses, 'recall@10': val_recalls}
    
    # Plot Training BCE Loss
    plt.figure()
    for (lr, wd), res in results.items():
        plt.plot(res['train_loss'], label=f'LR={lr}, WD={wd}')
    plt.title("Training BCE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot Validation Recall@10
    plt.figure()
    for (lr, wd), res in results.items():
        plt.plot(res['recall@10'], label=f'LR={lr}, WD={wd}')
    plt.title("Validation Recall@10")
    plt.xlabel("Epoch")
    plt.ylabel("Recall@10")
    plt.legend()
    plt.show()
    
    
    best_config = max(results.items(), key=lambda x: x[1]['recall@10'][-1])[0]
    best_lr, best_wd = best_config
    print(f"\nBest config: LR={best_lr}, WD={best_wd}")
    best_model = NMF(num_users, num_items, latent_dim=50).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=best_lr, weight_decay=best_wd)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(10), desc="Retraining Best Model", ncols=100):
        best_model.train()
        for user, item, label in train_loader:
            user, item, label = user.to(device), item.to(device), label.to(device)
            optimizer.zero_grad()
            preds = best_model(user, item)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

    # Do full evaluation on best model
    metrics = full_evaluate(best_model, val_df, train_df, num_items, device)
    print("\nFull Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")