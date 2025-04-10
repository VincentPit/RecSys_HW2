import pandas as pd
import pickle
import time

start_time = time.time()
ratings = pd.read_csv('../ml-20m/ratings.csv')
ratings = ratings.sort_values(by=['userId', 'timestamp'])

train_data = []
val_data = []
test_data = []

for userId, group in ratings.groupby('userId'):
    group = group.reset_index(drop=True)
    if len(group) >= 3:
        train_data.append(group.iloc[:-2])      # all but last two
        val_data.append(group.iloc[[-2]])       # second to last
        test_data.append(group.iloc[[-1]])      # last
    elif len(group) == 2:
        train_data.append(group.iloc[[0]])      # first
        test_data.append(group.iloc[[1]])       # last
    else:
        train_data.append(group)             

train_df = pd.concat(train_data, ignore_index=True)
val_df = pd.concat(val_data, ignore_index=True)
test_df = pd.concat(test_data, ignore_index=True)

with open("train.pkl", "wb") as f:
    pickle.dump(train_df, f)

with open("val.pkl", "wb") as f:
    pickle.dump(val_df, f)

with open("test.pkl", "wb") as f:
    pickle.dump(test_df, f)

elapsed_time = time.time() - start_time
print(f"Leave-One-Last Split Completed in {elapsed_time:.2f} seconds")
print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
