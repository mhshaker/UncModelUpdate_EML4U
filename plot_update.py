import numpy as np
import matplotlib.pyplot as plt


run_name = "sim data 50d 5c run10/"
# run_name = ""
run_name = "amazon data/"

with open(f"results/{run_name}stream_score_unc_list.npy", 'rb') as f:
    unc = np.load(f)
with open(f"results/{run_name}stream_score_score_list.npy", 'rb') as f:
    err = np.load(f)
with open(f"results/{run_name}stream_score_all_list.npy", 'rb') as f:
    all = np.load(f)
with open(f"results/{run_name}stream_score_no_list.npy", 'rb') as f:
    nou = np.load(f)


with open(f"results/{run_name}stream_count_unc_list.npy", 'rb') as f:
    unc_c = np.load(f)

with open(f"results/{run_name}stream_count_score_list.npy", 'rb') as f:
    err_c = np.load(f)

print(unc.shape)
unc_c = int(unc_c.mean())
err_c = int(err_c.mean())

unc = unc.mean(axis=0)[:-2]
err = err.mean(axis=0)[:-2]
all = all.mean(axis=0)[:-2]
nou = nou.mean(axis=0)[:-2]

print(unc.shape)
# unc = unc[0,:]
# err = err[0,:]
# all = all[0,:]

plt.plot(nou, label="nou (0)", alpha=0.8)
plt.plot(err, label=f"error ({err_c})", alpha=0.8)
plt.plot(unc, label=f"unc ({unc_c})", c="black", alpha=0.8)
plt.plot(all, label=f"all ({len(all)})", alpha=0.8)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Matthews correlation coefficient (MCC)")
plt.savefig(f"results/{run_name}_MCC.png")
plt.close()

exit()

plt.plot(nou[:,0], label="nou (0)", alpha=0.8) # , '--'
plt.plot(err[:,0], label=f"error ({err_c})", alpha=0.8)
plt.plot(unc[:,0], label=f"unc ({unc_c})", c="black", alpha=0.8)
plt.plot(all[:,0], label=f"all ({len(all)})", alpha=0.8)
plt.vlines(x=31, ymin=nou[:,0].min() - 0.1 * all[:,0].max(), ymax=all[:,0].max() + 0.1 * all[:,0].max(), color='gray', alpha=0.5)
plt.vlines(x=64, ymin=nou[:,0].min() - 0.1 * all[:,0].max(), ymax=all[:,0].max() + 0.1 * all[:,0].max(), color='gray', alpha=0.5)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Accuracy")
plt.savefig(f"results/{run_name}_ACC.png")
plt.close()

plt.plot(nou[:,1], label="nou (0)", alpha=0.8)
plt.plot(err[:,1], label=f"error ({err_c})", alpha=0.8)
plt.plot(unc[:,1], label=f"unc ({unc_c})", c="black", alpha=0.8)
plt.plot(all[:,1], label=f"all ({len(all)})", alpha=0.8)
plt.vlines(x=31, ymin=nou[:,1].min() - 0.1 * all[:,1].max(), ymax=all[:,1].max() + 0.1 * all[:,1].max(), color='gray', alpha=0.5)
plt.vlines(x=64, ymin=nou[:,1].min() - 0.1 * all[:,1].max(), ymax=all[:,1].max() + 0.1 * all[:,1].max(), color='gray', alpha=0.5)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Log_loss")
plt.savefig(f"results/{run_name}_Loss.png")
plt.close()

plt.plot(nou[:,2], label="nou (0)", alpha=0.8)
plt.plot(err[:,2], label=f"error ({err_c})", alpha=0.8)
plt.plot(unc[:,2], label=f"unc ({unc_c})", c="black", alpha=0.8)
plt.plot(all[:,2], label=f"all ({len(all)})", alpha=0.8)
plt.vlines(x=31, ymin=nou[:,2].min() - 0.1 * all[:,2].max(), ymax=all[:,2].max() + 0.1 * all[:,2].max(), color='gray', alpha=0.5)
plt.vlines(x=64, ymin=nou[:,2].min() - 0.1 * all[:,2].max(), ymax=all[:,2].max() + 0.1 * all[:,2].max(), color='gray', alpha=0.5)

plt.legend()
plt.xlabel("Episodes")
plt.ylabel("Matthews correlation coefficient (MCC)")
plt.savefig(f"results/{run_name}_MCC.png")
plt.close()
