import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from a import MovieLensSequenceDataset  
from b import SASRec  


max_seq_len = 50
hidden_dim = 50
batch_size = 128
num_heads = 1
num_epochs = 100
lr = 0.001
dropout = 0.2


torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

def sampled_eval(model, test_data, all_items, top_k=10, num_neg=100):
    model.eval()
    hits, ndcgs = [], []

    for user_seq, ground_truth in test_data:
        negatives = list(set(random.sample(all_items, num_neg)) - set(user_seq))
        sampled_items = negatives[:num_neg - 1] + [ground_truth]
        item_seq = torch.tensor(user_seq).unsqueeze(0).to(model.device)
        items = torch.tensor(sampled_items).unsqueeze(0).to(model.device)

        with torch.no_grad():
            logits = model(None, item_seq)  # shape (1, num_items)
            scores = logits[0][items[0]]
            rank = scores.argsort(descending=True)
            rank_index = (rank == (len(sampled_items) - 1)).nonzero(as_tuple=True)[0].item()

            hits.append(int(rank_index < top_k))
            ndcgs.append(1 / np.log2(rank_index + 2))

    return np.mean(hits), np.mean(ndcgs)
def full_eval(model, test_data, all_items_tensor, top_k=10):
    model.eval()
    hits, ndcgs = [], []

    for user_seq, ground_truth in test_data:
        item_seq = torch.tensor(user_seq).unsqueeze(0).to(model.device)
        with torch.no_grad():
            logits = model(None, item_seq)  # shape (1, num_items)
            scores = logits[0]
            rank = scores.argsort(descending=True)
            rank_index = (rank == ground_truth).nonzero(as_tuple=True)[0].item()
            hits.append(int(rank_index < top_k))
            ndcgs.append(1 / np.log2(rank_index + 2))

    return np.mean(hits), np.mean(ndcgs)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

dataset = MovieLensSequenceDataset(
        ratings_path="../ml-20m/ratings.csv",
        max_len=max_seq_len,
        split="train"
    )
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

val_dataset = MovieLensSequenceDataset(
    ratings_path="../ml-20m/ratings.csv",
    max_len=max_seq_len,
    split="val",
    user2id=dataset.user2id,
    item2id=dataset.item2id
)

test_dataset = MovieLensSequenceDataset(
    ratings_path="../ml-20m/ratings.csv",
    max_len=max_seq_len,
    split="test",
    user2id=dataset.user2id,
    item2id=dataset.item2id
)

all_items = list(range(len(dataset.item2id)))

# === Build Model ===
model = SASRec(
    num_users=len(dataset.user2id),
    num_items=len(dataset.item2id),
    max_seq_len=max_seq_len,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout
).to(device)


# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        user_seq = batch['seq'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(None, user_seq)  # (B, T, num_items)
        logits = logits[:, -1, :]       # last position
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}")

    hr, ndcg = sampled_eval(model, val_dataset.eval_tuples, all_items, top_k=10)
    print(f"[Validation] HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

# === Final Test Evaluation ===
print("\nRunning Final Test Evaluation...")
hr, ndcg = sampled_eval(model, test_dataset.eval_tuples, all_items, top_k=10)
print(f"[Sampled Test] HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

# === Bonus: Full Evaluation ===
full_hr, full_ndcg = full_eval(model, test_dataset.eval_tuples, torch.tensor(all_items).to(device), top_k=10)
print(f"[Full Test] HR@10: {full_hr:.4f}, NDCG@10: {full_ndcg:.4f}")

# === Save Results ===
torch.save(model.state_dict(), "sasrec_model.pth")