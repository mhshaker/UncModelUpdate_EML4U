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


# Parameters
episodes_p =  0.05
seed = 1
# load data

# features_all, targets_all   = dp.load_data("./Data/")
# data = np.concatenate((features_all,targets_all), axis=1)

with open('Data/amazon_small.npy', 'rb') as f:
    data = np.load(f)

# data = pd.read_csv('Data/spambase.csv')
# data = data.sample(frac=1)

stream = DataStream(data)
stream_length = stream.n_remaining_samples()
episode_size = int(stream_length * episodes_p)

stream_score_unc_list = []
stream_score_score_list = []
stream_score_all_list = []

mode = "run all"

for seed in range(5):
    print("------------------------------------ seed ", seed)
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
        update_model = True
        updatecounter = 0
        episode = 0
        stream_score_unc = []
        while True:
            # setup new episode
            x_ep, y_ep = stream.next_sample(episode_size)
            if len(y_ep) == 0:
                break
            episode +=1

            s = matthews_corrcoef(y_ep, model.predict(x_ep))
            # s = model.score(x_ep, y_ep)
            stream_score_unc.append(s)

            tu, eu, au = unc.model_uncertainty(model, x_ep, x_train, y_train, laplace_smoothing=1)
            tu_ep = tu.mean()
            if tu_ep > tu_set:
                update_model = True
            #     print("episode ", episode, "D")
            # else:
            #     print("episode ", episode)

            # train the model / update
            if update_model:
                x_train, y_train = x_ep, y_ep
                model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
                model.fit(x_ep, y_ep) # remove keys when fiting the model
                update_model = False
                updatecounter +=1
        stream_score_unc_list.append(stream_score_unc)
        print("unc update count", updatecounter)

        unc_save = np.array(stream_score_unc_list)
        with open('results/stream_score_unc_list.npy', 'wb') as f:
            np.save(f, unc_save)

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

            s = matthews_corrcoef(y_ep, model.predict(x_ep))
            # s = model.score(x_ep, y_ep)
            stream_score_score.append(s)

            if s > score_set:
                update_model = True
            #     print("episode ", episode, "D")
            # else:
            #     print("episode ", episode)


            # train the model / update
            if update_model:
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

    if mode == "all" or mode == "run all":
        # always update
        stream.restart()
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        x_train, y_train = stream.next_sample(episode_size)
        model.fit(x_train, y_train)

        _, _ = stream.next_sample(episode_size * 2)

        # episode loop
        update_model = True
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

            s = matthews_corrcoef(y_ep, model.predict(x_ep))
            stream_score_all.append(s)

            # train the model / update
            if update_model:
                x_train, y_train = x_ep, y_ep
                model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
                model.fit(x_ep, y_ep) # remove keys when fiting the model
                updatecounter_all +=1
        stream_score_all_list.append(stream_score_all)
        print("all update count", updatecounter_all)

        all_save = np.array(stream_score_all_list)
        with open('results/stream_score_all_list.npy', 'wb') as f:
            np.save(f, all_save)

    if mode == "no" or mode == "run all":
        # always update
        stream.restart()
        model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=seed)
        x_train, y_train = stream.next_sample(episode_size)
        model.fit(x_train, y_train)

        _, _ = stream.next_sample(episode_size * 2)

        # episode loop
        update_model = True
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

            s = matthews_corrcoef(y_ep, model.predict(x_ep))
            stream_score_all.append(s)

        stream_score_all_list.append(stream_score_all)
        print("all update count", updatecounter_all)

        all_save = np.array(stream_score_all_list)
        with open('results/stream_score_no_list.npy', 'wb') as f:
            np.save(f, all_save)
