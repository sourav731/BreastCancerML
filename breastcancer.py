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

# # VISUALISATIAN
# 

# %%
def boxplotFloatDataTypes(columnName):
    plt.figure(figsize=(10,10))
    sns.boxplot(x='diagnosis',y=columnName,data=df)
    plt.show()


# %%
# for col in df.columns:
#     if col != 'diagnosis' and col != 'id':
#         boxplotFloatDataTypes(col)
#     else :
#         continue


# %%
def linePlot(columnName):
    plt.figure(figsize=(25,5))
    plt.plot(df[columnName],marker='o',label=columnName)
    plt.legend()
    meanArray = []
    xAxis = []
    for i in range(1,570):
        meanArray.append(np.mean(df[columnName]))
        xAxis.append(i)

    plt.plot(xAxis,meanArray,color='red',linestyle='dashed',label='mean')
    plt.legend()


# %%
# for col in df.columns:
#     if col != 'diagnosis' and col != 'id':
#         linePlot(col)
#     else :
#         continue


# %%
# df.skew()


# %%
# df['concavity_se'].skew()


# %%
# df['area_se'].skew()


# %%
# df['concavity_se'].plot()


# %%
# df[df['concavity_se'] > 0.20]


# %%
# df['area_se'].plot()


# %%
# df[df['area_se'] > 200]


# %%
df['area_se'] = np.log(df['area_se'])


# %%
# df['concavity_se']


# %%
df['concavity_se'] = np.log(df['concavity_se'])


# %%
df['concavity_se'] = df['concavity_se'].replace([np.inf, -np.inf], 0)


# %%
# df['concavity_se'].skew()


# %%
# df['radius_se'].skew()
# df['radius_se'] = np.log(df['radius_se'])
