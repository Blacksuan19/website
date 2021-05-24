---
title: Mobile Price Range Classification Via Lazy Predict
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/mobile-price-range-classification-via-lazy-predict
tags:
  - data-science
  - machine-learning
  - project
---

lazy predict is a library that trains a large number of models on a given dataset to determine which one will work best for it

the goal is to predict a price range for a smartphone based on its specifications.

the specifcations include a total of 20 columns ranging from 3g availability to touch screen and amount of ram so a very extensive feature set.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
```

```python
train = pd.read_csv("../input/mobile-price-classification/train.csv")
test = pd.read_csv("../input/mobile-price-classification/test.csv")
```

after loading in the data, lets take a look at it

```python
train.head()
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
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>n_cores</th>
      <th>...</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
      <th>price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>842</td>
      <td>0</td>
      <td>2.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>0.6</td>
      <td>188</td>
      <td>2</td>
      <td>...</td>
      <td>20</td>
      <td>756</td>
      <td>2549</td>
      <td>9</td>
      <td>7</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1021</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>53</td>
      <td>0.7</td>
      <td>136</td>
      <td>3</td>
      <td>...</td>
      <td>905</td>
      <td>1988</td>
      <td>2631</td>
      <td>17</td>
      <td>3</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>563</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>41</td>
      <td>0.9</td>
      <td>145</td>
      <td>5</td>
      <td>...</td>
      <td>1263</td>
      <td>1716</td>
      <td>2603</td>
      <td>11</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>615</td>
      <td>1</td>
      <td>2.5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0.8</td>
      <td>131</td>
      <td>6</td>
      <td>...</td>
      <td>1216</td>
      <td>1786</td>
      <td>2769</td>
      <td>16</td>
      <td>8</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1821</td>
      <td>1</td>
      <td>1.2</td>
      <td>0</td>
      <td>13</td>
      <td>1</td>
      <td>44</td>
      <td>0.6</td>
      <td>141</td>
      <td>2</td>
      <td>...</td>
      <td>1208</td>
      <td>1212</td>
      <td>1411</td>
      <td>8</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>

```python
test.head()
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
      <th>id</th>
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>...</th>
      <th>pc</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1043</td>
      <td>1</td>
      <td>1.8</td>
      <td>1</td>
      <td>14</td>
      <td>0</td>
      <td>5</td>
      <td>0.1</td>
      <td>193</td>
      <td>...</td>
      <td>16</td>
      <td>226</td>
      <td>1412</td>
      <td>3476</td>
      <td>12</td>
      <td>7</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>841</td>
      <td>1</td>
      <td>0.5</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>61</td>
      <td>0.8</td>
      <td>191</td>
      <td>...</td>
      <td>12</td>
      <td>746</td>
      <td>857</td>
      <td>3895</td>
      <td>6</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1807</td>
      <td>1</td>
      <td>2.8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>27</td>
      <td>0.9</td>
      <td>186</td>
      <td>...</td>
      <td>4</td>
      <td>1270</td>
      <td>1366</td>
      <td>2396</td>
      <td>17</td>
      <td>10</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1546</td>
      <td>0</td>
      <td>0.5</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>25</td>
      <td>0.5</td>
      <td>96</td>
      <td>...</td>
      <td>20</td>
      <td>295</td>
      <td>1752</td>
      <td>3893</td>
      <td>10</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1434</td>
      <td>0</td>
      <td>1.4</td>
      <td>0</td>
      <td>11</td>
      <td>1</td>
      <td>49</td>
      <td>0.5</td>
      <td>108</td>
      <td>...</td>
      <td>18</td>
      <td>749</td>
      <td>810</td>
      <td>1773</td>
      <td>15</td>
      <td>8</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>

## Data Analysis

```python
# check data types
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2000 entries, 0 to 1999
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype
    ---  ------         --------------  -----
     0   battery_power  2000 non-null   int64
     1   blue           2000 non-null   int64
     2   clock_speed    2000 non-null   float64
     3   dual_sim       2000 non-null   int64
     4   fc             2000 non-null   int64
     5   four_g         2000 non-null   int64
     6   int_memory     2000 non-null   int64
     7   m_dep          2000 non-null   float64
     8   mobile_wt      2000 non-null   int64
     9   n_cores        2000 non-null   int64
     10  pc             2000 non-null   int64
     11  px_height      2000 non-null   int64
     12  px_width       2000 non-null   int64
     13  ram            2000 non-null   int64
     14  sc_h           2000 non-null   int64
     15  sc_w           2000 non-null   int64
     16  talk_time      2000 non-null   int64
     17  three_g        2000 non-null   int64
     18  touch_screen   2000 non-null   int64
     19  wifi           2000 non-null   int64
     20  price_range    2000 non-null   int64
    dtypes: float64(2), int64(19)
    memory usage: 328.2 KB

```python
# check if there are any null columns
train.isna().sum()
```

    battery_power    0
    blue             0
    clock_speed      0
    dual_sim         0
    fc               0
    four_g           0
    int_memory       0
    m_dep            0
    mobile_wt        0
    n_cores          0
    pc               0
    px_height        0
    px_width         0
    ram              0
    sc_h             0
    sc_w             0
    talk_time        0
    three_g          0
    touch_screen     0
    wifi             0
    price_range      0
    dtype: int64

```python
# describe the data
train.describe()
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
      <th>battery_power</th>
      <th>blue</th>
      <th>clock_speed</th>
      <th>dual_sim</th>
      <th>fc</th>
      <th>four_g</th>
      <th>int_memory</th>
      <th>m_dep</th>
      <th>mobile_wt</th>
      <th>n_cores</th>
      <th>...</th>
      <th>px_height</th>
      <th>px_width</th>
      <th>ram</th>
      <th>sc_h</th>
      <th>sc_w</th>
      <th>talk_time</th>
      <th>three_g</th>
      <th>touch_screen</th>
      <th>wifi</th>
      <th>price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2000.000000</td>
      <td>2000.0000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>...</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1238.518500</td>
      <td>0.4950</td>
      <td>1.522250</td>
      <td>0.509500</td>
      <td>4.309500</td>
      <td>0.521500</td>
      <td>32.046500</td>
      <td>0.501750</td>
      <td>140.249000</td>
      <td>4.520500</td>
      <td>...</td>
      <td>645.108000</td>
      <td>1251.515500</td>
      <td>2124.213000</td>
      <td>12.306500</td>
      <td>5.767000</td>
      <td>11.011000</td>
      <td>0.761500</td>
      <td>0.503000</td>
      <td>0.507000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>439.418206</td>
      <td>0.5001</td>
      <td>0.816004</td>
      <td>0.500035</td>
      <td>4.341444</td>
      <td>0.499662</td>
      <td>18.145715</td>
      <td>0.288416</td>
      <td>35.399655</td>
      <td>2.287837</td>
      <td>...</td>
      <td>443.780811</td>
      <td>432.199447</td>
      <td>1084.732044</td>
      <td>4.213245</td>
      <td>4.356398</td>
      <td>5.463955</td>
      <td>0.426273</td>
      <td>0.500116</td>
      <td>0.500076</td>
      <td>1.118314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>501.000000</td>
      <td>0.0000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.100000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>500.000000</td>
      <td>256.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>851.750000</td>
      <td>0.0000</td>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>0.200000</td>
      <td>109.000000</td>
      <td>3.000000</td>
      <td>...</td>
      <td>282.750000</td>
      <td>874.750000</td>
      <td>1207.500000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1226.000000</td>
      <td>0.0000</td>
      <td>1.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>32.000000</td>
      <td>0.500000</td>
      <td>141.000000</td>
      <td>4.000000</td>
      <td>...</td>
      <td>564.000000</td>
      <td>1247.000000</td>
      <td>2146.500000</td>
      <td>12.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1615.250000</td>
      <td>1.0000</td>
      <td>2.200000</td>
      <td>1.000000</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>48.000000</td>
      <td>0.800000</td>
      <td>170.000000</td>
      <td>7.000000</td>
      <td>...</td>
      <td>947.250000</td>
      <td>1633.000000</td>
      <td>3064.500000</td>
      <td>16.000000</td>
      <td>9.000000</td>
      <td>16.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1998.000000</td>
      <td>1.0000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>19.000000</td>
      <td>1.000000</td>
      <td>64.000000</td>
      <td>1.000000</td>
      <td>200.000000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>1960.000000</td>
      <td>1998.000000</td>
      <td>3998.000000</td>
      <td>19.000000</td>
      <td>18.000000</td>
      <td>20.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>

# Explortary Data Analaysis

```python
# number of samples for each price range
fig, ax = plt.subplots(figsize = (10, 4))
sns.countplot(x ='price_range', data=train)
plt.xlabel("Class Label")
plt.ylabel("Number of Samples")
plt.show()
```

![png](/assets/images/mobile-price-range-classification-via-lazy-predict_files/mobile-price-range-classification-via-lazy-predict_11_0.png)

perfectly balanced, as all things should be.

```python
# find correlation
corr_mat = train.corr()

# each columns correlation with the price
corr_mat['price_range']
```

    battery_power    0.200723
    blue             0.020573
    clock_speed     -0.006606
    dual_sim         0.017444
    fc               0.021998
    four_g           0.014772
    int_memory       0.044435
    m_dep            0.000853
    mobile_wt       -0.030302
    n_cores          0.004399
    pc               0.033599
    px_height        0.148858
    px_width         0.165818
    ram              0.917046
    sc_h             0.022986
    sc_w             0.038711
    talk_time        0.021859
    three_g          0.023611
    touch_screen    -0.030411
    wifi             0.018785
    price_range      1.000000
    Name: price_range, dtype: float64

```python
# convert all to positive and sort by value
abs(corr_mat).sort_values(by=['price_range'])['price_range']
```

    m_dep            0.000853
    n_cores          0.004399
    clock_speed      0.006606
    four_g           0.014772
    dual_sim         0.017444
    wifi             0.018785
    blue             0.020573
    talk_time        0.021859
    fc               0.021998
    sc_h             0.022986
    three_g          0.023611
    mobile_wt        0.030302
    touch_screen     0.030411
    pc               0.033599
    sc_w             0.038711
    int_memory       0.044435
    px_height        0.148858
    px_width         0.165818
    battery_power    0.200723
    ram              0.917046
    price_range      1.000000
    Name: price_range, dtype: float64

we can make a few observations from above

- the ram is the most deciding factor in price range with the highest correlation.
- the amount of pixels do matter after all.
- number of cores does not correlate with the price much (could be due to the cores being weak, for example most midrangers nowadays have 8 cores while the Apple A series SoCs have at most 6 cores and still perform miles better).

```python
# battery correlation plot
fig, ax = plt.subplots(figsize=(14,10))
sns.boxenplot(x="price_range",y="battery_power", data=train,ax = ax)
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa979ba8dd0>

![png](/assets/images/mobile-price-range-classification-via-lazy-predict_files/mobile-price-range-classification-via-lazy-predict_16_1.png)

```python
# individual correlation graphs

# get all columns and remove price_range
cols = list(train.columns.values)
cols.remove('price_range')

# plot figure
fig, ax = plt.subplots(7, 3, figsize=(15, 30))
plt.subplots_adjust(left=0.1, bottom=0.05, top=1.0, wspace=0.3, hspace=0.2)
for i, col in zip(range(len(cols)), cols):
    ax = plt.subplot(7,3,i+1)
    sns.lineplot(ax=ax,x='price_range', y=col, data=train)
```

![png](/assets/images/mobile-price-range-classification-via-lazy-predict_files/mobile-price-range-classification-via-lazy-predict_17_0.png)

```python
# plot full heatmap
figure(figsize=(20, 14))
sns.heatmap(corr_mat, annot = True, fmt='.1g', cmap= 'coolwarm')
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa9771fd850>

![png](/assets/images/mobile-price-range-classification-via-lazy-predict_files/mobile-price-range-classification-via-lazy-predict_18_1.png)

# Modeling

knowing which model to build for a dataset is not an easy task, specially when the columns that have a high correlation with the target variable are less than half the total columns, its also a task that is time consuming in making and tuning these models that is why we will use the LazyPredict library to show us the results of various models without any tuneing and we will implement the top 3 models.

```python
# extract target column
target = train['price_range']

# drop target column from dataset
train.drop('price_range', axis=1, inplace=True)
```

```python
from sklearn.model_selection import train_test_split

# install and import lazypredict
!pip install lazypredict
from lazypredict.Supervised import LazyClassifier

# split training dataset to training and testing
X_train, X_test, y_train, y_test = train_test_split(train, target,test_size=.3,random_state =123)

# make Lazyclassifier model(s)
lazy_clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# fit model(s)
models, predictions = lazy_clf.fit(X_train, X_test, y_train, y_test)
```

    Requirement already satisfied: lazypredict in /opt/conda/lib/python3.7/site-packages (0.2.7)
    Requirement already satisfied: Click>=7.0 in /opt/conda/lib/python3.7/site-packages (from lazypredict) (7.1.1)
    [33mWARNING: You are using pip version 20.3.1; however, version 20.3.3 is available.
    You should consider upgrading via the '/opt/conda/bin/python3.7 -m pip install --upgrade pip' command.[0m


    100%|██████████| 30/30 [00:04<00:00,  6.21it/s]

```python
models
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
      <th>Accuracy</th>
      <th>Balanced Accuracy</th>
      <th>ROC AUC</th>
      <th>F1 Score</th>
      <th>Time Taken</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LogisticRegression</th>
      <td>0.94</td>
      <td>0.95</td>
      <td>None</td>
      <td>0.94</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>LinearDiscriminantAnalysis</th>
      <td>0.93</td>
      <td>0.93</td>
      <td>None</td>
      <td>0.93</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>QuadraticDiscriminantAnalysis</th>
      <td>0.92</td>
      <td>0.92</td>
      <td>None</td>
      <td>0.92</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>LGBMClassifier</th>
      <td>0.91</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.91</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>XGBClassifier</th>
      <td>0.91</td>
      <td>0.91</td>
      <td>None</td>
      <td>0.90</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>RandomForestClassifier</th>
      <td>0.87</td>
      <td>0.87</td>
      <td>None</td>
      <td>0.87</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>SVC</th>
      <td>0.86</td>
      <td>0.87</td>
      <td>None</td>
      <td>0.86</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>NuSVC</th>
      <td>0.86</td>
      <td>0.87</td>
      <td>None</td>
      <td>0.86</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>BaggingClassifier</th>
      <td>0.86</td>
      <td>0.87</td>
      <td>None</td>
      <td>0.86</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>ExtraTreesClassifier</th>
      <td>0.84</td>
      <td>0.85</td>
      <td>None</td>
      <td>0.84</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>DecisionTreeClassifier</th>
      <td>0.84</td>
      <td>0.84</td>
      <td>None</td>
      <td>0.84</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>LinearSVC</th>
      <td>0.82</td>
      <td>0.83</td>
      <td>None</td>
      <td>0.82</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>CalibratedClassifierCV</th>
      <td>0.81</td>
      <td>0.82</td>
      <td>None</td>
      <td>0.80</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>GaussianNB</th>
      <td>0.78</td>
      <td>0.79</td>
      <td>None</td>
      <td>0.78</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>PassiveAggressiveClassifier</th>
      <td>0.74</td>
      <td>0.75</td>
      <td>None</td>
      <td>0.74</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>SGDClassifier</th>
      <td>0.73</td>
      <td>0.74</td>
      <td>None</td>
      <td>0.72</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>Perceptron</th>
      <td>0.73</td>
      <td>0.74</td>
      <td>None</td>
      <td>0.73</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>NearestCentroid</th>
      <td>0.70</td>
      <td>0.70</td>
      <td>None</td>
      <td>0.70</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>AdaBoostClassifier</th>
      <td>0.63</td>
      <td>0.62</td>
      <td>None</td>
      <td>0.61</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>RidgeClassifier</th>
      <td>0.56</td>
      <td>0.58</td>
      <td>None</td>
      <td>0.47</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>RidgeClassifierCV</th>
      <td>0.56</td>
      <td>0.58</td>
      <td>None</td>
      <td>0.47</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>BernoulliNB</th>
      <td>0.53</td>
      <td>0.54</td>
      <td>None</td>
      <td>0.52</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>KNeighborsClassifier</th>
      <td>0.52</td>
      <td>0.52</td>
      <td>None</td>
      <td>0.52</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>ExtraTreeClassifier</th>
      <td>0.51</td>
      <td>0.51</td>
      <td>None</td>
      <td>0.51</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>LabelSpreading</th>
      <td>0.45</td>
      <td>0.45</td>
      <td>None</td>
      <td>0.45</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>LabelPropagation</th>
      <td>0.45</td>
      <td>0.45</td>
      <td>None</td>
      <td>0.45</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>DummyClassifier</th>
      <td>0.26</td>
      <td>0.26</td>
      <td>None</td>
      <td>0.26</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>CheckingClassifier</th>
      <td>0.25</td>
      <td>0.25</td>
      <td>None</td>
      <td>0.10</td>
      <td>0.02</td>
    </tr>
  </tbody>
</table>
</div>

```python
# plot the first 5 models F1 score
top = models[:5]
figure(figsize=(14, 7))
sns.lineplot(x=top.index, y="F1 Score", data=top)
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fa946d44bd0>

![png](/assets/images/mobile-price-range-classification-via-lazy-predict_files/mobile-price-range-classification-via-lazy-predict_23_1.png)

we are not really intrested in the predictions dataframe here because we already know those values and they're part of the training dataset

from above we can see that the best algorithm for this type of task is logistic regression followed by Discriminant Analysis models and followed closely by GB models.

### Implemented models

- logistic regression
- Linear Discriminant Analysis
- light GBM classifier

the reason behing skipping on the Quadratic Discriminant Analysis model is because its of the same family as Linear Discriminant Analysis and produces similar results, we also want to implement a diverse range of models

```python
from sklearn.linear_model import LogisticRegression
# Logistic regression
log_clf = LogisticRegression(random_state=0).fit(train, target)
```

```python
# drop the id column from test to match the size of train
test.drop('id', axis=1, inplace=True)
```

```python
# get predictions on test dataset and convert it to a dataframe
log_preds = pd.DataFrame(log_clf.predict(test), columns = ['log_price_range'])

log_preds.head()
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
      <th>log_price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Linear Discriminant Analysis
lda_clf = LinearDiscriminantAnalysis().fit(train, target)
```

```python
# get predictions on test dataset and convert it to a dataframe
lda_preds = pd.DataFrame(lda_clf.predict(test), columns = ['lda_price_range'])

lda_preds.head()
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
      <th>lda_price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

```python
from lightgbm import LGBMClassifier
# lightgbm model
lgbm_clf = LGBMClassifier(objective='multiclass', random_state=5).fit(train, target)
```

```python
# get predictions on test dataset and convert it to a dataframe
lgbm_preds = pd.DataFrame(lgbm_clf.predict(test), columns = ['lgbm_price_range'])

lgbm_preds.head()
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
      <th>lgbm_price_range</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

### comparing model results

```python
# create dataframe with 3 columns and index from any of the predicted dataframes
results = pd.DataFrame(index=log_preds.index, columns=['log', 'lda', 'lgbm'])

# add in data from the 3 predicted dfs
results['log'] = log_preds
results['lda'] = lda_preds
results['lgbm'] = lgbm_preds

# show grouped df
results
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
      <th>log</th>
      <th>lda</th>
      <th>lgbm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>996</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>999</th>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 3 columns</p>
</div>

```python
# find columns where all 3 models agree on the result
equal_rows = 0
for row in results.itertuples(index=False):
    if(row.log == row.lda == row.lgbm):
        equal_rows += 1

equal_rows
```

    628

from all the 1000 rows the 3 models agree on 62% which means any of these 3 algorithms should be n overall good choice for predicting the price range of a smartphone based on its specifications
