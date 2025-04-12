import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MovieLensSequenceDataset(Dataset):
    def __init__(self, interactions, num_items, max_seq_len=50):
        """
        interactions: list of (user, item) tuples
        num_items: total number of items
        max_seq_len: maximum sequence length
        """
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.user_interactions = self._build_user_interactions(interactions)
        self.sequences = self._create_sequences()
        print("Sequential dataset built.")

    def _build_user_interactions(self, interactions):
        df = pd.DataFrame(interactions, columns=['user', 'item'])
        df = df.sort_values(['user'])  # optional, can add timestamp if available
        user_seq = df.groupby('user')['item'].apply(list).to_dict()
        return user_seq

    def _create_sequences(self):
        np.random.seed(1)
        sequences = []

        for user, item_list in self.user_interactions.items():
            if len(item_list) < 2:
                continue  # not enough history for sequence

            for i in range(1, len(item_list)):
                seq = item_list[max(0, i - self.max_seq_len):i]
                target = item_list[i]

                # Sample 1 negative item not in user history
                neg_item = np.random.randint(0, self.num_items)
                while neg_item in item_list:
                    neg_item = np.random.randint(0, self.num_items)

                sequences.append((user, seq, target, neg_item))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        user, seq, pos_item, neg_item = self.sequences[idx]

        # Padding
        padded_seq = np.zeros(self.max_seq_len, dtype=np.int64)
        padded_seq[-len(seq):] = seq

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )
        
        
if __name__ == "__main__":
    ratings_file = "../ml-20m/ratings.csv"
    ratings_df = pd.read_csv(ratings_file)

    interactions = ratings_df[['userId', 'movieId']].drop_duplicates()
    user2id = {uid: i for i, uid in enumerate(interactions['userId'].unique())}
    item2id = {iid: i for i, iid in enumerate(interactions['movieId'].unique())}
    interactions['userId'] = interactions['userId'].map(user2id)
    interactions['movieId'] = interactions['movieId'].map(item2id)

    num_users = len(user2id)
    num_items = len(item2id)

    interaction_tuples = list(zip(interactions['userId'], interactions['movieId']))

    seq_dataset = MovieLensSequenceDataset(interactions=interaction_tuples, num_items=num_items, max_seq_len=10)
    dataloader = DataLoader(seq_dataset, batch_size=4, shuffle=True)

    for user, seq, pos_item, neg_item in dataloader:
        print("User:", user)
        print("Sequence:", seq)
        print("Pos Item:", pos_item)
        print("Neg Item:", neg_item)
        break

