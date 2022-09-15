import numpy as np
import matplotlib.pyplot as plt


with open('results/stream_score_unc_list.npy', 'rb') as f:
    unc = np.load(f)
with open('results/stream_score_unc_list.npy', 'rb') as f:
    err = np.load(f)
with open('results/stream_score_unc_list.npy', 'rb') as f:
    all = np.load(f)
with open('results/stream_score_no_list.npy', 'rb') as f:
    nou = np.load(f)

print(unc.shape)
unc = unc.mean(axis=0)
err = err.mean(axis=0)
all = all.mean(axis=0)
nou = nou.mean(axis=0)
# unc = unc[0,:]
# err = err[0,:]
# all = all[0,:]

plt.plot(nou, label="nou")
plt.plot(unc, label="unc")
plt.plot(err, label="error")
plt.plot(all, label="all")
plt.legend()
plt.savefig("results/plot.png")
