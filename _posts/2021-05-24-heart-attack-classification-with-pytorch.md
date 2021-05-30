---
title: Heart Attack Classification With Pytorch
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/heart-attack-classification-with-pytorch
tags:
  - data-science
  - machine-learning
  - project
---

```python
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# set default figure size
plt.rcParams['figure.figsize'] = (15, 7.0)
```

```python
heart_data = '../input/heart-attack-analysis-prediction-dataset/heart.csv'

heart_df = pd.read_csv(heart_data)

heart_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
# describe the data
heart_df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
      <td>303.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>54.366337</td>
      <td>0.683168</td>
      <td>0.966997</td>
      <td>131.623762</td>
      <td>246.264026</td>
      <td>0.148515</td>
      <td>0.528053</td>
      <td>149.646865</td>
      <td>0.326733</td>
      <td>1.039604</td>
      <td>1.399340</td>
      <td>0.729373</td>
      <td>2.313531</td>
      <td>0.544554</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.082101</td>
      <td>0.466011</td>
      <td>1.032052</td>
      <td>17.538143</td>
      <td>51.830751</td>
      <td>0.356198</td>
      <td>0.525860</td>
      <td>22.905161</td>
      <td>0.469794</td>
      <td>1.161075</td>
      <td>0.616226</td>
      <td>1.022606</td>
      <td>0.612277</td>
      <td>0.498835</td>
    </tr>
    <tr>
      <th>min</th>
      <td>29.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>94.000000</td>
      <td>126.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>47.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>120.000000</td>
      <td>211.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>133.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>55.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>130.000000</td>
      <td>240.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>153.000000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>61.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>140.000000</td>
      <td>274.500000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>166.000000</td>
      <td>1.000000</td>
      <td>1.600000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>77.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>200.000000</td>
      <td>564.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>202.000000</td>
      <td>1.000000</td>
      <td>6.200000</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
# checking data types
heart_df.dtypes
```

    age           int64
    sex           int64
    cp            int64
    trtbps        int64
    chol          int64
    fbs           int64
    restecg       int64
    thalachh      int64
    exng          int64
    oldpeak     float64
    slp           int64
    caa           int64
    thall         int64
    output        int64
    dtype: object

```python
# drop duplicates if any
heart_df.drop_duplicates()

# check missing valus
heart_df.isna().sum()
```

    age         0
    sex         0
    cp          0
    trtbps      0
    chol        0
    fbs         0
    restecg     0
    thalachh    0
    exng        0
    oldpeak     0
    slp         0
    caa         0
    thall       0
    output      0
    dtype: int64

```python
# check output column class distribution
sns.countplot(x='output', data=heart_df).set_title("output Column Distribution")
```

    Text(0.5, 1.0, 'output Column Distribution')

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_5_1.png)

```python
# check sex column class distribution
sns.countplot(x='sex', data=heart_df).set_title("Sex Column Distribution")
```

    Text(0.5, 1.0, 'Sex Column Distribution')

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_6_1.png)

```python
# box plot for output and cholestrol level
sns.boxplot(x="output",y="chol",data=heart_df)
```

    <AxesSubplot:xlabel='output', ylabel='chol'>

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_7_1.png)

```python
# box plot for output and cholestrol level
sns.boxplot(x="output",y="thalachh",data=heart_df)
```

    <AxesSubplot:xlabel='output', ylabel='thalachh'>

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_8_1.png)

```python
# box plot for output and cholestrol level
sns.boxplot(x="output",y="oldpeak",data=heart_df)
```

    <AxesSubplot:xlabel='output', ylabel='oldpeak'>

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_9_1.png)

```python
# box plot for output and cholestrol level
sns.boxplot(x="output",y="age",data=heart_df)
```

    <AxesSubplot:xlabel='output', ylabel='age'>

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_10_1.png)

```python
ax = sns.countplot(x='age', data=heart_df)
```

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_11_0.png)

```python
# check correlation
corr = heart_df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set_title("Columns Correlation")
```

    Text(0.5, 1.0, 'Columns Correlation')

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_12_1.png)

```python
# split data for training
y = heart_df.output.to_numpy()
X = heart_df.drop('output', axis=1).to_numpy()

# scale X values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# split data while keeping output class distribution consistent
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
```

```python
# convert data to pytorch tensors
def df_to_tensor(df):
    return torch.from_numpy(df).float()

X_traint = df_to_tensor(X_train)
y_traint = df_to_tensor(y_train)
X_testt = df_to_tensor(X_test)
y_testt = df_to_tensor(y_test)
```

```python
# create pytorch dataset
train_ds = TensorDataset(X_traint, y_traint)
test_ds = TensorDataset(X_testt, y_testt)

# create data loaders
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)
```

```python
# model architecture
class BinaryNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.out(x)
        return torch.sigmoid(x) # scaling values between 0 and 1
```

```python
input_size = 13 # number of features
output_size = 1
model = BinaryNetwork(input_size, output_size)
loss_fn = nn.BCELoss() # Binary Cross Entropy
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
model
```

    BinaryNetwork(
      (l1): Linear(in_features=13, out_features=64, bias=True)
      (l2): Linear(in_features=64, out_features=32, bias=True)
      (l3): Linear(in_features=32, out_features=16, bias=True)
      (out): Linear(in_features=16, out_features=1, bias=True)
    )

```python
epochs = 100
losses = []
for i in range(epochs):
    epoch_loss = 0
    for feat, target in train_dl:
        optim.zero_grad()
        out = model(feat)
        loss = loss_fn(out, target.unsqueeze(1))
        epoch_loss += loss.item()
        loss.backward()
        optim.step()
    losses.append(epoch_loss)
    # print loss every 10
    if i % 10 == 0:
        print(f"Epoch: {i}/{epochs}, Loss = {loss:.5f}")
```

    Epoch: 0/100, Loss = 0.79641
    Epoch: 10/100, Loss = 0.03637
    Epoch: 20/100, Loss = 0.07704
    Epoch: 30/100, Loss = 0.02023
    Epoch: 40/100, Loss = 0.00084
    Epoch: 50/100, Loss = 0.00000
    Epoch: 60/100, Loss = 0.00001
    Epoch: 70/100, Loss = 0.00000
    Epoch: 80/100, Loss = 0.00018
    Epoch: 90/100, Loss = 0.00029

```python
# plot losses
graph = sns.lineplot(x=[x for x in range(0, epochs)], y=losses)
graph.set(title="Loss change during training", xlabel='epochs', ylabel='loss')
plt.show()
```

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_19_0.png)

```python
# evaluate the model
y_pred_list = []
model.eval()
with torch.no_grad():
    for X, y in test_dl:
        y_test_pred = model(X)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag)

# convert predictions to a list of tensors with 1 dimention
y_pred_list = [a.squeeze() for a in y_pred_list]
```

```python
# check confusion matrix (hstack will merge all tensor lists into one list)
cfm = confusion_matrix(y_test, torch.hstack(y_pred_list))
sns.heatmap(cfm / np.sum(cfm), annot=True, fmt='.2%')
```

    <AxesSubplot:>

![png](/assets/images/heart-attack-classification-with-pytorch_files/heart-attack-classification-with-pytorch_21_1.png)

```python
# print metrics
print(classification_report(y_test, torch.hstack(y_pred_list)))
```

                  precision    recall  f1-score   support

               0       0.91      0.75      0.82        28
               1       0.82      0.94      0.87        33

        accuracy                           0.85        61
       macro avg       0.86      0.84      0.85        61
    weighted avg       0.86      0.85      0.85        61
