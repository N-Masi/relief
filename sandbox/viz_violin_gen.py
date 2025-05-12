import matplotlib.pyplot as plt
import pandas as pd
import pickle

with open('results/gdf_territory_data_ckpt.pkl', 'rb') as f: 
    td = pickle.load(f)
df = pd.DataFrame(td)

# sort data
df = df[df['model'] == 'keisler']
df = df[df['lead_time'] == '12']
df = df[df['variable'] == 'T850']

plt.clf()
plt.violinplot(df.rmse)
plt.boxplot(df.rmse)
plt.xticks(None)
plt.ylabel('RMSE')
plt.title('RMSE of T850 for each territory with Keisler model, 12h lead time')
plt.savefig('untracked/territory_violin_keisler12h_t850.png')
