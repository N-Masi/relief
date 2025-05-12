import plotly.express as px
import pandas as pd
import pickle

with open('results/gdf_income_data_ckpt.pkl', 'rb') as f:
    income = pickle.load(f)

df_income = pd.DataFrame(income)
df_income.lead_time = pd.to_numeric(df_income.lead_time)
df_income = df_income.rename(columns={'subregion': 'income'})
fig = px.line(
    df_income[df_income['variable']=='T850'], 
    x='lead_time', y='rmse', 
    color='income', 
    facet_col='model',
    title='RMSE of T850 by Model of Income Groups'
)
fig.write_html("untracked/income_plotly.html")
