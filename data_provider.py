import numpy as np
from amazon_pickle_reader import AmazonPickleReader

def load_data(data_address, text_key=False):
    data = AmazonPickleReader(data_address)
    features = data.get_all_bow50()["data"][0] # 0 gives the embeddings and 1 gives the metadata
    if text_key:
        targets  = np.array(data.get_all_bow50()["data"][1])[:,[1, 4]] # colomn index 1 is the class labels and colomn index 4 the text keys
    else:
        targets  = np.array(data.get_all_bow50()["data"][1])[:,[1]] # colomn index 1 is the class labels
        # targets = targets.reshape(-1)

    

    return features, targets.astype(int)

def partition_data(features, targets, partition_number): # to create episodes
    features_list = []
    targets_list = []

    partition_len = int(len(features) / partition_number)

    for i in range(partition_number):
        features_list.append(features[i*partition_len: (i+1) * partition_len])
        targets_list.append(targets[i*partition_len: (i+1) * partition_len])
    return features_list, targets_list

