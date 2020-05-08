# # LOAD DATA

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# %%
df = pd.read_csv('data.csv')


# %%
# df.head()


# %%
# df.describe()


# %%
# df.shape


# %%
# df.columns


# %%
# df.info()


# %%
df = df.drop(['Unnamed: 32'],axis=1)


# %%
# df


# %%
# df['diagnosis'].value_counts()
