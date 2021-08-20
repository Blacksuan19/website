---
title: Credit Card Fraud Detection With Pytorch
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/credit-card-fraud-detection-with-pytorch_files/cover.png
source: https://www.kaggle.com/abubakaryagob/credit-card-fraud-detection-with-pytorch
tags:
  - data-science
  - machine-learning
  - project
---

This notebook classifies credit card transactions to fraudulent or non
fraudulent, the dataset is a set of PCA features extracted from the original
data in order to conceal the identities of the parties in question.

```python
# install torchsummary
!pip install -q torchsummary
```

    [33mWARNING: Running pip as root will break packages and permissions. You should install packages reliably by using venv: https://pip.pypa.io/warnings/venv[0m

```python
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import sum
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchsummary import summary

# set figure size
plt.rcParams["figure.figsize"] = (14,7)
```

```python
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>-0.966272</td>
      <td>-0.185226</td>
      <td>1.792993</td>
      <td>-0.863291</td>
      <td>-0.010309</td>
      <td>1.247203</td>
      <td>0.237609</td>
      <td>0.377436</td>
      <td>-1.387024</td>
      <td>...</td>
      <td>-0.108300</td>
      <td>0.005274</td>
      <td>-0.190321</td>
      <td>-1.175575</td>
      <td>0.647376</td>
      <td>-0.221929</td>
      <td>0.062723</td>
      <td>0.061458</td>
      <td>123.50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>-1.158233</td>
      <td>0.877737</td>
      <td>1.548718</td>
      <td>0.403034</td>
      <td>-0.407193</td>
      <td>0.095921</td>
      <td>0.592941</td>
      <td>-0.270533</td>
      <td>0.817739</td>
      <td>...</td>
      <td>-0.009431</td>
      <td>0.798278</td>
      <td>-0.137458</td>
      <td>0.141267</td>
      <td>-0.206010</td>
      <td>0.502292</td>
      <td>0.219422</td>
      <td>0.215153</td>
      <td>69.99</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>

```python
# some stats about the data
print(f"Number of data points: {df.size}")
print(f"Number of Fradulant Transactions: {df['Class'].value_counts()[1]}")
print(f"Number of non-fradulant Transactions: {df['Class'].value_counts()[0]}\n\n")
sns.countplot(x=df["Class"], palette="YlGnBu").set(title="Class Balance Between Transcation Types")
plt.show()
```

    Number of data points: 8829017
    Number of Fradulant Transactions: 492
    Number of non-fradulant Transactions: 284315

![png](/assets/images/credit-card-fraud-detection-with-pytorch_files/credit-card-fraud-detection-with-pytorch_4_1.png)

there is huge class impalance in the data, this might lead to a biased model, we
can mitigate this by only using the same amount of class 0 while training or we
could generate some sample data from the given features.

```python
# Amount per transaction for each type
sns.scatterplot(data=df.reset_index(), x="index", y="Amount", hue="Class", cmap="YlGnBu").set(title="Amount per transaction")
plt.show()
```

![png](/assets/images/credit-card-fraud-detection-with-pytorch_files/credit-card-fraud-detection-with-pytorch_6_0.png)

fraudulent transactions dont tend to have a large sum of cash per transaction,
we can confirm this by calculating some statistics such as max, min and mean for
each type of transaction.

```python
for i, word in zip(range(2), ["Positive", "Negative"]):
    print(f"{word} transactions statistics")
    print(df[df["Class"] == i]["Amount"].describe(), "\n\n")
```

    Positive transactions statistics
    count    284315.000000
    mean         88.291022
    std         250.105092
    min           0.000000
    25%           5.650000
    50%          22.000000
    75%          77.050000
    max       25691.160000
    Name: Amount, dtype: float64


    Negative transactions statistics
    count     492.000000
    mean      122.211321
    std       256.683288
    min         0.000000
    25%         1.000000
    50%         9.250000
    75%       105.890000
    max      2125.870000
    Name: Amount, dtype: float64

```python
# split data into training and testing
X = df.drop("Class", axis=1)
y = df["Class"]

# scale the values of x (better training)
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y) # stratify keeps class balance
```

```python
# create tensor datasets from df
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train.values)
y_test = torch.FloatTensor(y_test.values)
train_ds = TensorDataset(X_train, y_train)
```

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
```

    cuda

```python
# create dataloaders
batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size)
```

```python
# Network Architecture
class FraudNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # make the number of hidden dim layers configurable
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())

        # final layer
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out = self.input(x)
        for layer in self.layers:
            out = layer(out)
        return self.fc(out)
```

```python
# training function
def train_model(model, epochs, loss_fn, optimizer):
    model.train()
    for epoch in range(epochs):
        with tqdm(train_dl, unit="batch") as tepoch:
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                preds = model(data)
                loss = loss_fn(preds, target.long())
                loss.backward()
                optimizer.step()
                tepoch.set_postfix(loss=loss.item())
```

```python
inp_size = X_train.shape[1]
model = FraudNet(inp_size, inp_size).to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),  lr = 1e-4)

# summarize the model layers
summary(model, (inp_size, inp_size))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1               [-1, 30, 30]             930
                  ReLU-2               [-1, 30, 30]               0
                Linear-3               [-1, 30, 30]             930
                  ReLU-4               [-1, 30, 30]               0
                Linear-5               [-1, 30, 30]             930
                  ReLU-6               [-1, 30, 30]               0
                Linear-7               [-1, 30, 30]             930
                  ReLU-8               [-1, 30, 30]               0
                Linear-9               [-1, 30, 30]             930
                 ReLU-10               [-1, 30, 30]               0
               Linear-11                [-1, 30, 2]              62
    ================================================================
    Total params: 4,712
    Trainable params: 4,712
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.07
    Params size (MB): 0.02
    Estimated Total Size (MB): 0.09
    ----------------------------------------------------------------

```python
epochs = 10
train_model(model, epochs, loss, optim)
```

    Epoch 0: 100%|██████████| 1558/1558 [00:11<00:00, 137.25batch/s, loss=0.000822]
    Epoch 1: 100%|██████████| 1558/1558 [00:11<00:00, 138.96batch/s, loss=0.000756]
    Epoch 2: 100%|██████████| 1558/1558 [00:11<00:00, 139.36batch/s, loss=0.0022]
    Epoch 3: 100%|██████████| 1558/1558 [00:11<00:00, 133.62batch/s, loss=0.00293]
    Epoch 4: 100%|██████████| 1558/1558 [00:11<00:00, 137.87batch/s, loss=0.00276]
    Epoch 5: 100%|██████████| 1558/1558 [00:11<00:00, 137.02batch/s, loss=0.00241]
    Epoch 6: 100%|██████████| 1558/1558 [00:11<00:00, 135.01batch/s, loss=0.00206]
    Epoch 7: 100%|██████████| 1558/1558 [00:11<00:00, 139.75batch/s, loss=0.0018]
    Epoch 8: 100%|██████████| 1558/1558 [00:11<00:00, 134.36batch/s, loss=0.00159]
    Epoch 9: 100%|██████████| 1558/1558 [00:11<00:00, 138.24batch/s, loss=0.00143]

```python
model.eval()
preds = model(X_test.to("cuda")).argmax(dim=1)
print(classification_report(y_test, preds.cpu()))
```

                  precision    recall  f1-score   support

             0.0       1.00      1.00      1.00     85295
             1.0       0.90      0.76      0.82       148

        accuracy                           1.00     85443
       macro avg       0.95      0.88      0.91     85443
    weighted avg       1.00      1.00      1.00     85443

```python
class_def = {0 : "Not Fraudulent", 1 : "Fraudulent"}
cm_df = pd.DataFrame(confusion_matrix(y_test, preds.cpu())).rename(columns=class_def, index=class_def)
cm_df = cm_df / sum(cm_df)
sns.heatmap(cm_df, annot=True, fmt='0.2%', cmap="YlGnBu").set(title="Confusion Matrix", xlabel="Predicted Label", ylabel="True Label")
plt.show()
```

![png](/assets/images/credit-card-fraud-detection-with-pytorch_files/credit-card-fraud-detection-with-pytorch_18_0.png)

as expected the model does have some issue with classifiying fraudulent
transactions, this can be addressed in multiple ways:

- use the same amount of data for both classes
- generate more data for fraudulent class 1
- use a deeper network (more layers)
- use a different network architecture
- use other algorithms
