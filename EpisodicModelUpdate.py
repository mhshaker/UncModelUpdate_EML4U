# load data -- Done
# train model 
# episode update loop
    # seperate data to episods -- Done
    # uncertainty calculation
    # update code


import data_provider as dp
import numpy as np
from skmultiflow.data import DataStream
from sklearn.ensemble import RandomForestClassifier
import Uncertainty as unc
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss
from sklearn.datasets import make_classification
import math

# Parameters
episodes_p =  0.01
runs = 10
data_name = "sim"
n_samples = 50000
n_classes = 5
########################################################### load data

if data_name == "amazon":
    features_all, targets_all   = dp.load_data("./Data/")
    data = np.concatenate((features_all,targets_all), axis=1)
elif data_name == "amazon_small":
    with open('Data/amazon_small.npy', 'rb') as f:
        data = np.load(f)

########################################################### metrics

def measures(model, x_ep, y_ep):
    mcc = matthews_corrcoef(y_ep, model.predict(x_ep))
    acc = model.score(x_ep, y_ep)
    loss = log_loss(y_ep,model.predict_proba(x_ep))
    return [acc, loss, mcc]

stream_score_unc_list = []
stream_score_score_list = []
stream_score_all_list = []
stream_score_nou_list = []

stream_count_unc_list = []
stream_count_score_list = []


mode = "run all"

for seed in range(runs):
    print("------------------------------------ run ", seed)

    if data_name == "sim": ########################## sim data
        x_n, y_n = make_classification(n_samples=n_samples * 2, n_features=500, n_informative=300, n_redundant=5, 
                                n_repeated=0, n_classes=n_classes, n_clusters_per_class=1, weights=None, 
                                flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                                shuffle=True, random_state=seed)

        x_d, y_d = make_classification(n_samples=n_samples,     n_features=500, n_informative=300, n_redundant=15, 
                                n_repeated=0, n_classes=n_classes, n_clusters_per_class=1, weights=None, 
                                flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, 
                                shuffle=True, random_state=seed+1)

        x_ndn = np.concatenate((x_n[:n_samples], x_d, x_n[n_samples:]), axis=0)
        y_ndn = np.concatenate((y_n[:n_samples], y_d, y_n[n_samples:]), axis=0)
        data = np.concatenate((x_ndn,y_ndn.reshape(-1,1)), axis=1) 

    stream = DataStream(data)
    stream_length = stream.n_remaining_samples()
    episode_size = int(stream_length * episodes_p)

    if data_name == "sim":
        drift_episode1 = math.ceil(n_samples/episode_size)
        drift_episode2 = math.ceil(n_samples*2/episode_size)

    # defining the model
    if mode == "unc" or mode == "run all":
        stream.restart()
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        x_train, y_train = stream.next_sample(episode_size)
        model.fit(x_train, y_train)
        # set uncertainty score
        x_set, y_set = stream.next_sample(episode_size * 2)
        tu, eu, au = unc.model_uncertainty(model, x_set, x_train, y_train, laplace_smoothing=1)

        tu_set = tu.mean()
        # Uncertainty detection
        # episode loop
        updatecounter_unc = 0
        episode = 0
        stream_score_unc = []
        while True:
            # setup new episode
            x_ep, y_ep = stream.next_sample(episode_size)
            if len(y_ep) == 0:
                break
            episode +=1

            # s = matthews_corrcoef(y_ep, model.predict(x_ep))
            # s = model.score(x_ep, y_ep)
            m = measures(model, x_ep, y_ep)
            stream_score_unc.append(m)

            tu, eu, au = unc.model_uncertainty(model, x_ep, x_train, y_train, laplace_smoothing=1)
            tu_ep = tu.mean()
            if tu_ep > tu_set:
                x_train, y_train = x_ep, y_ep
                model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
                model.fit(x_ep, y_ep) # remove keys when fiting the model
                updatecounter_unc +=1
            #     print("episode ", episode, "D")
            # else:
            #     print("episode ", episode)

            # train the model / update
        stream_score_unc_list.append(stream_score_unc)
        print("unc update count", updatecounter_unc)

        unc_save = np.array(stream_score_unc_list)
        print(unc_save.shape)
        with open('results/stream_score_unc_list.npy', 'wb') as f:
            np.save(f, unc_save)

        stream_count_unc_list.append(updatecounter_unc)
        unc_save_c = np.array(stream_count_unc_list)
        with open('results/stream_count_unc_list.npy', 'wb') as f:
            np.save(f, unc_save_c)

    if mode == "err" or mode == "run all":

        # Error detection
        stream.restart()
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        x_train, y_train = stream.next_sample(episode_size)
        model.fit(x_train, y_train)

        x_set, y_set = stream.next_sample(episode_size * 2)
        # score_set = model.score(x_set, y_set)
        score_set = matthews_corrcoef(y_set, model.predict(x_set))
        # episode loop
        update_model = True
        updatecounter_score = 0
        episode = 0
        stream_score_score = []
        while True:
            # setup new episode
            x_ep, y_ep = stream.next_sample(episode_size)
            if len(y_ep) == 0:
                break
            episode +=1

            # s = matthews_corrcoef(y_ep, model.predict(x_ep))
            # s = model.score(x_ep, y_ep)
            m = measures(model, x_ep, y_ep)
            stream_score_score.append(m)

            # train the model / update
            if m[2] < score_set: # score_set is MCC - higher is better
                x_train, y_train = x_ep, y_ep
                model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
                model.fit(x_ep, y_ep) # remove keys when fiting the model
                update_model = False
                updatecounter_score +=1

        stream_score_score_list.append(stream_score_score)
        print("score update count", updatecounter_score)

        score_save = np.array(stream_score_score_list)
        with open('results/stream_score_score_list.npy', 'wb') as f:
            np.save(f, score_save)

        stream_count_score_list.append(updatecounter_score)
        score_save_c = np.array(stream_count_score_list)
        with open('results/stream_count_score_list.npy', 'wb') as f:
            np.save(f, score_save_c)

    if mode == "all" or mode == "run all":
        # always update
        stream.restart()
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        x_train, y_train = stream.next_sample(episode_size)
        model.fit(x_train, y_train)

        _, _ = stream.next_sample(episode_size * 2)

        # episode loop
        updatecounter_all = 0
        episode = 0
        stream_score_all = []
        while True:
            # print("episode ", episode)
            # setup new episode
            x_ep, y_ep = stream.next_sample(episode_size)
            if len(y_ep) == 0:
                break
            episode +=1

            # s = matthews_corrcoef(y_ep, model.predict(x_ep))
            m = measures(model, x_ep, y_ep)
            stream_score_all.append(m)

            # train the model / update
            x_train, y_train = x_ep, y_ep
            model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
            model.fit(x_ep, y_ep) # remove keys when fiting the model
            updatecounter_all +=1

        stream_score_all_list.append(stream_score_all)
        print("all update count", updatecounter_all)

        all_save = np.array(stream_score_all_list)
        with open('results/stream_score_all_list.npy', 'wb') as f:
            np.save(f, all_save)

    if mode == "nou" or mode == "run all":
        # always update
        stream.restart()
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        x_train, y_train = stream.next_sample(episode_size)
        model.fit(x_train, y_train)

        _, _ = stream.next_sample(episode_size * 2)

        # episode loop
        updatecounter_no = 0
        episode = 0
        stream_score_nou = []
        while True:
            # print("episode ", episode)
            # setup new episode
            x_ep, y_ep = stream.next_sample(episode_size)
            if len(y_ep) == 0:
                break
            episode +=1

            # s = matthews_corrcoef(y_ep, model.predict(x_ep))
            m = measures(model, x_ep, y_ep)
            stream_score_nou.append(m)

        stream_score_nou_list.append(stream_score_nou)
        print("all update count", updatecounter_no)

        nou_save = np.array(stream_score_nou_list)
        with open('results/stream_score_no_list.npy', 'wb') as f:
            np.save(f, nou_save)
