import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MovieLensTripletDataset(Dataset):
    def __init__(self, interactions, num_items, neg_k=1):
        """
        interactions: list of (user_id, item_id)
        num_items: total number of unique items
        neg_k: number of negative samples per positive
        """
        self.neg_k = neg_k
        self.num_items = num_items
        self.interactions_df = pd.DataFrame(interactions, columns=['user', 'item'])

        
        self.user_pos_items = self.interactions_df.groupby('user')['item'].agg(set).to_dict()
        print("built user")
        # Cache the triplets: (user, pos_item, neg_item)
        self.triplets = self._create_triplets()
        print("cache triplets")

    def _create_triplets(self):
        np.random.seed(1)
        users = []
        pos_items = []
        neg_items = []

        user_list = list(self.user_pos_items.keys())
        for user in user_list:
            pos_item_list = list(self.user_pos_items[user])
            pos_item_arr = np.array(pos_item_list)

            # Repeat each positive item neg_k times
            repeated_pos_items = np.repeat(pos_item_arr, self.neg_k)
            repeated_users = np.repeat(user, len(repeated_pos_items))

            # Efficient negative sampling (rejection-based sampling vectorized)
            neg_sample_pool = np.random.randint(0, self.num_items, size=len(repeated_pos_items) * 2)
            mask = ~np.isin(neg_sample_pool, pos_item_arr)
            valid_neg_samples = neg_sample_pool[mask][:len(repeated_pos_items)]

            # Append to lists
            users.append(repeated_users)
            pos_items.append(repeated_pos_items)
            neg_items.append(valid_neg_samples)

        return list(zip(
            np.concatenate(users),
            np.concatenate(pos_items),
            np.concatenate(neg_items)
        ))

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return tuple(map(torch.tensor, self.triplets[idx]))



if __name__ == "__main__":
    # Set the file path to your ratings.csv from the ml-20m dataset
    ratings_file = "../ml-20m/ratings.csv"

    # Load ratings and keep only userId and movieId (implicit feedback)
    ratings_df = pd.read_csv(ratings_file)
    print("Raw data loaded.")

    # Filter implicit interactions (all non-zero ratings are treated as positive)
    interactions = ratings_df[['userId', 'movieId']].drop_duplicates()

    # Map userId and movieId to indices starting from 0
    user2id = {uid: i for i, uid in enumerate(interactions['userId'].unique())}
    item2id = {iid: i for i, iid in enumerate(interactions['movieId'].unique())}

    # Apply mapping
    interactions['userId'] = interactions['userId'].map(user2id)
    interactions['movieId'] = interactions['movieId'].map(item2id)

    num_users = len(user2id)
    num_items = len(item2id)

    interaction_tuples = list(zip(interactions['userId'], interactions['movieId']))

    print(f"Total users: {num_users}, Total items: {num_items}")
    print(f"Total positive interactions: {len(interaction_tuples)}")

   
    neg_k = 2  # 2 negative samples per positive
    dataset = MovieLensTripletDataset(interactions=interaction_tuples, num_items=num_items, neg_k=neg_k)
    print("Dataset created.")
    
    # Wrap in a DataLoader if needed
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Data loader created.")
    # Print a few samples
    print("Sample triplets (user, pos_item, neg_item):")
    for batch in dataloader:
        for triplet in zip(*batch):
            print(triplet)
        break 