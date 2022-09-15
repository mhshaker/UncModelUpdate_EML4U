import pandas as pd
import data_provider as dp
import numpy as np

# features_all, targets_all   = dp.load_data("./Data/")
# data = np.concatenate((features_all,targets_all), axis=1)

# data_small = data[:10000]

# with open('Data/amazon_small.npy', 'wb') as f:
#     np.save(f, data_small)

with open('Data/amazon_small.npy', 'rb') as f:
    data_small = np.load(f)

print(data_small.shape)