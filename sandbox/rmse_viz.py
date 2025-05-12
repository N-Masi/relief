import matplotlib.pyplot as plt
import pickle
import pdb

with open('untracked/240x121/neuralgcm-ens_rmses.pkl', 'rb') as f:
    rmses = pickle.load(f)

fig, ax = plt.subplots(2)

ax[0].scatter(rmses['lead_times'], y=rmses['T850']['oceans'], c='blue', label='oceans')
ax[0].scatter(rmses['lead_times'], y=rmses['T850']['land'], c='green', label='land')
ax[0].scatter(rmses['lead_times'], y=rmses['T850']['lakes'], c='gray', label='lakes')
ax[0].set_xlabel('lead time (hours)')
ax[0].set_ylabel('RMSE')
ax[0].set_title("NeuralGCM (Ensemble) RMSE of T850 (temporally averged over 2020)")
ax[0].legend()

ax[1].scatter(rmses['lead_times'], y=rmses['Z500']['oceans'], c='blue', label='oceans')
ax[1].scatter(rmses['lead_times'], y=rmses['Z500']['land'], c='green', label='land')
ax[1].scatter(rmses['lead_times'], y=rmses['Z500']['lakes'], c='gray', label='lakes')
ax[1].set_xlabel('lead time (hours)')
ax[1].set_ylabel('RMSE')
ax[1].set_title("NeuralGCM (Ensemble) RMSE of Z500 (temporally averaged over 2020)")
ax[1].legend()

fig.tight_layout()
plt.savefig('untracked/240x121/NeuralGCM-ens.png')
