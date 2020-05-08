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

# df['concavity_se'].skew()

# df['radius_se'].skew()
# df['radius_se'] = np.log(df['radius_se'])

# ### BI VARIATE

# DROP HIGHLY CO RELATED COLUMNS
df = df.drop(['perimeter_mean','area_mean','concave points_mean','radius_worst','perimeter_worst','area_worst','texture_worst', 'smoothness_worst', 'compactness_worst','concavity_worst', 'concave points_worst', 'symmetry_worst','fractal_dimension_worst','compactness_mean','perimeter_se','area_se','texture_se'],axis=1)


# %%
df['diagnosis'] = df['diagnosis'].map({'M':1,'B':0})


# BI VARIATE ANALYSIS
# df.corr()


# %%
# ax = sns.scatterplot(x='radius_mean',y='texture_mean',data=df)


# %%
# ax = sns.scatterplot(x='radius_mean',y='smoothness_mean',data=df)


# %%
# ax = sns.scatterplot(x='radius_mean',y='concavity_mean',data=df)


# %%
# ax = sns.scatterplot(x='radius_mean',y='smoothness_se',data=df)


# %%
# ax = sns.scatterplot(x="radius_mean", y="texture_mean",hue="smoothness_mean",sizes=(20, 200),data=df)

# %% [markdown]
# ## MODEL

# MODELING
from sklearn.model_selection import train_test_split


#creating training and testing vars
df = df.drop(['id'],axis=1)
train,test = train_test_split(df,test_size=0.1,shuffle=True)

# df.head()


# TRAIN AND TEST SET CREATION
y_train = train['diagnosis']
x_train = train.drop(['diagnosis'],axis=1)

y_test = test['diagnosis']
x_test = test.drop(['diagnosis'],axis=1)


# IMPORT MODEL
from sklearn.linear_model import LogisticRegression


# MODEL TRAINED
clf = LogisticRegression(random_state=0).fit(x_train,y_train)


# SCORE
# clf.score(x_test,y_test)

## PICKEL
import pickle

filename = 'breastcancer.pkl'
with open(filename, 'wb') as file: 
    pickle.dump(clf, file)


# %%
with open(filename,'rb') as file:
    model = pickle.load(file)


# %%
model.score(x_test,y_test)
