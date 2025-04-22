import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MovieLensSequenceDataset(Dataset):
    def __init__(
        self,
        ratings_path=None,
        interactions=None,
        max_seq_len=50,
        split="train",
        user2id=None,
        item2id=None,
        test_ratio=0.1,
        val_ratio=0.1
    ):
        """
        Either `ratings_path` or `interactions` must be provided.
        """

        assert ratings_path is not None or interactions is not None, "Need ratings_path or interactions"

        self.max_seq_len = max_seq_len
        self.split = split.lower()

        if ratings_path:
            ratings_df = pd.read_csv(ratings_path)
        else:
            #userId,movieId,rating,timestamp
            ratings_df = pd.DataFrame(interactions, columns=["userId", "movieId"])

        ratings_df = ratings_df.sort_values(["userId", "timestamp"] if "timestamp" in ratings_df.columns else ["user"])

        # Create user and item mappings
        if user2id is None:
            self.user2id = {u: i for i, u in enumerate(ratings_df["userId"].unique())}
        else:
            self.user2id = user2id

        if item2id is None:
            self.item2id = {i: j for j, i in enumerate(ratings_df["movieId"].unique())}
        else:
            self.item2id = item2id

        ratings_df["user"] = ratings_df["userId"].map(self.user2id)
        ratings_df["item"] = ratings_df["movieId"].map(self.item2id)

        self.num_items = len(self.item2id)
        self.user_interactions = ratings_df.groupby("user")["item"].apply(list).to_dict()

        self.sequences = self._create_sequences(split, val_ratio, test_ratio)
        print(f"{split.capitalize()} dataset built: {len(self.sequences)} sequences")

    def _create_sequences(self, split, val_ratio, test_ratio):
        sequences = []
        np.random.seed(1)

        for user, item_list in self.user_interactions.items():
            if len(item_list) < 3:
                continue

            n_total = len(item_list)
            n_test = int(n_total * test_ratio)
            n_val = int(n_total * val_ratio)

            if split == "train":
                item_range = range(1, n_total - n_val - n_test)
            elif split == "val":
                item_range = range(n_total - n_val - n_test, n_total - n_test)
            elif split == "test":
                item_range = range(n_total - n_test, n_total)
            else:
                raise ValueError("split must be 'train', 'val', or 'test'")

            for i in item_range:
                seq = item_list[max(0, i - self.max_seq_len):i]
                target = item_list[i]

                neg_item = np.random.randint(self.num_items)
                while neg_item in item_list:
                    neg_item = np.random.randint(self.num_items)

                sequences.append((user, seq, target, neg_item))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        user, seq, pos_item, neg_item = self.sequences[idx]
        padded_seq = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_seq[-len(seq):] = seq

        return user, padded_seq, pos_item, neg_item
        
        
if __name__ == "__main__":
    ratings_path = "../ml-20m/ratings.csv"
    max_seq_len = 50
    batch_size = 1024
    
    train_dataset = MovieLensSequenceDataset(
        ratings_path=ratings_path,
        max_seq_len=max_seq_len,
        split="train"
    )

    val_dataset = MovieLensSequenceDataset(
        ratings_path=ratings_path,
        max_seq_len=max_seq_len,
        split="val",
        user2id=train_dataset.user2id,
        item2id=train_dataset.item2id
    )

    test_dataset = MovieLensSequenceDataset(
        ratings_path=ratings_path,
        max_seq_len=max_seq_len,
        split="test",
        user2id=train_dataset.user2id,
        item2id=train_dataset.item2id
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)