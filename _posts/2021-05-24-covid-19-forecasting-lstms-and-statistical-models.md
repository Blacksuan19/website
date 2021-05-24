---
title: Covid 19 Forecasting Lstms And Statistical Models
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/covid-19-forecasting-lstms-and-statistical-models
tags:
  - data-science
  - machine-learning
  - project
---

# Introduction

Since we have already analyzed all these datasets in the target countries section, we see that using the global dataset for all our modeling is the best option for a few reasons:

- it contains data for all the countries we are covering
- it has up to date data that includes the whole lifetime of the pandemic
- the individual countries datasets are not as complete in some cases
- a single dataset is arguably easier to work with compared to many
- the data is already clean and
- we have already confirmed the creadability of the data

### Objectives

our main objective is to see which one of our chosen 4 countries have handled the virus in a way that can be generalized to everyone as simple guidelines, the targeted countries are

- United States
- Germany
- Italy
- South Korea

# Data Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, GlobalMaxPooling1D, Bidirectional
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

%matplotlib inline

# supress annoying warning
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
```

```python
df_confirmed = pd.read_csv("../input/covid-19/time_series_covid19_confirmed_global.csv")
df_deaths = pd.read_csv("../input/covid-19/time_series_covid19_deaths_global.csv")
df_reco = pd.read_csv("../input/covid-19/time_series_covid19_recovered_global.csv")
```

after reading in our dataset lets take a look at it by showing the first few countries for confirmed case, deaths, and recoveries

```python
df_confirmed.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>10/22/20</th>
      <th>10/23/20</th>
      <th>10/24/20</th>
      <th>10/25/20</th>
      <th>10/26/20</th>
      <th>10/27/20</th>
      <th>10/28/20</th>
      <th>10/29/20</th>
      <th>10/30/20</th>
      <th>10/31/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>40626</td>
      <td>40687</td>
      <td>40768</td>
      <td>40833</td>
      <td>40937</td>
      <td>41032</td>
      <td>41145</td>
      <td>41268</td>
      <td>41334</td>
      <td>41425</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>18250</td>
      <td>18556</td>
      <td>18858</td>
      <td>19157</td>
      <td>19445</td>
      <td>19729</td>
      <td>20040</td>
      <td>20315</td>
      <td>20634</td>
      <td>20875</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>55357</td>
      <td>55630</td>
      <td>55880</td>
      <td>56143</td>
      <td>56419</td>
      <td>56706</td>
      <td>57026</td>
      <td>57332</td>
      <td>57651</td>
      <td>57942</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3811</td>
      <td>4038</td>
      <td>4038</td>
      <td>4038</td>
      <td>4325</td>
      <td>4410</td>
      <td>4517</td>
      <td>4567</td>
      <td>4665</td>
      <td>4756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8582</td>
      <td>8829</td>
      <td>9026</td>
      <td>9381</td>
      <td>9644</td>
      <td>9871</td>
      <td>10074</td>
      <td>10269</td>
      <td>10558</td>
      <td>10805</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 288 columns</p>
</div>

```python
df_deaths.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>10/22/20</th>
      <th>10/23/20</th>
      <th>10/24/20</th>
      <th>10/25/20</th>
      <th>10/26/20</th>
      <th>10/27/20</th>
      <th>10/28/20</th>
      <th>10/29/20</th>
      <th>10/30/20</th>
      <th>10/31/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1505</td>
      <td>1507</td>
      <td>1511</td>
      <td>1514</td>
      <td>1518</td>
      <td>1523</td>
      <td>1529</td>
      <td>1532</td>
      <td>1533</td>
      <td>1536</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>465</td>
      <td>469</td>
      <td>473</td>
      <td>477</td>
      <td>480</td>
      <td>487</td>
      <td>493</td>
      <td>499</td>
      <td>502</td>
      <td>509</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1888</td>
      <td>1897</td>
      <td>1907</td>
      <td>1914</td>
      <td>1922</td>
      <td>1931</td>
      <td>1941</td>
      <td>1949</td>
      <td>1956</td>
      <td>1964</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>63</td>
      <td>69</td>
      <td>69</td>
      <td>69</td>
      <td>72</td>
      <td>72</td>
      <td>72</td>
      <td>73</td>
      <td>75</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>260</td>
      <td>265</td>
      <td>267</td>
      <td>268</td>
      <td>270</td>
      <td>271</td>
      <td>275</td>
      <td>275</td>
      <td>279</td>
      <td>284</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 288 columns</p>
</div>

```python
df_reco.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>10/22/20</th>
      <th>10/23/20</th>
      <th>10/24/20</th>
      <th>10/25/20</th>
      <th>10/26/20</th>
      <th>10/27/20</th>
      <th>10/28/20</th>
      <th>10/29/20</th>
      <th>10/30/20</th>
      <th>10/31/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>67.709953</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>33831</td>
      <td>34010</td>
      <td>34023</td>
      <td>34129</td>
      <td>34150</td>
      <td>34217</td>
      <td>34237</td>
      <td>34239</td>
      <td>34258</td>
      <td>34321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>20.168300</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>10395</td>
      <td>10466</td>
      <td>10548</td>
      <td>10654</td>
      <td>10705</td>
      <td>10808</td>
      <td>10893</td>
      <td>11007</td>
      <td>11097</td>
      <td>11189</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Algeria</td>
      <td>28.03390</td>
      <td>1.659600</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>38618</td>
      <td>38788</td>
      <td>38932</td>
      <td>39095</td>
      <td>39273</td>
      <td>39444</td>
      <td>39635</td>
      <td>39635</td>
      <td>40014</td>
      <td>40201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Andorra</td>
      <td>42.50630</td>
      <td>1.521800</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2470</td>
      <td>2729</td>
      <td>2729</td>
      <td>2729</td>
      <td>2957</td>
      <td>3029</td>
      <td>3144</td>
      <td>3260</td>
      <td>3377</td>
      <td>3475</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Angola</td>
      <td>-11.20270</td>
      <td>17.873900</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3305</td>
      <td>3384</td>
      <td>3461</td>
      <td>3508</td>
      <td>3530</td>
      <td>3647</td>
      <td>3693</td>
      <td>3736</td>
      <td>4107</td>
      <td>4523</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 288 columns</p>
</div>

after taking a look at the data as a whole lets now get our target countries each in their own dataframes

```python
us_confirmed = df_confirmed[df_confirmed["Country/Region"] == "US"]
us_deaths = df_deaths[df_deaths["Country/Region"] == "US"]
us_reco = df_reco[df_reco["Country/Region"] == "US"]

germany_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Germany"]
germany_deaths = df_deaths[df_deaths["Country/Region"] == "Germany"]
germany_reco = df_reco[df_reco["Country/Region"] == "Germany"]

italy_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Italy"]
italy_deaths = df_deaths[df_deaths["Country/Region"] == "Italy"]
italy_reco = df_reco[df_reco["Country/Region"] == "Italy"]

sk_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Korea, South"]
sk_deaths = df_deaths[df_deaths["Country/Region"] == "Korea, South"]
sk_reco = df_reco[df_reco["Country/Region"] == "Korea, South"]
```

```python
us_reco
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>10/22/20</th>
      <th>10/23/20</th>
      <th>10/24/20</th>
      <th>10/25/20</th>
      <th>10/26/20</th>
      <th>10/27/20</th>
      <th>10/28/20</th>
      <th>10/29/20</th>
      <th>10/30/20</th>
      <th>10/31/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>231</th>
      <td>NaN</td>
      <td>US</td>
      <td>40.0</td>
      <td>-100.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3353056</td>
      <td>3375427</td>
      <td>3406656</td>
      <td>3422878</td>
      <td>3460455</td>
      <td>3487666</td>
      <td>3518140</td>
      <td>3554336</td>
      <td>3578452</td>
      <td>3612478</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 288 columns</p>
</div>

with the current data structure shown above we cant do much so lets first convert it to a form that can used to make graphs or train a model

```python
## structuring timeseries data
def confirmed_timeseries(df):
    df_series = pd.DataFrame(df[df.columns[4:]].sum(),columns=["confirmed"])
    df_series.index = pd.to_datetime(df_series.index,format = '%m/%d/%y')
    return df_series

def deaths_timeseries(df):
    df_series = pd.DataFrame(df[df.columns[4:]].sum(),columns=["deaths"])
    df_series.index = pd.to_datetime(df_series.index,format = '%m/%d/%y')
    return df_series

def reco_timeseries(df):
    # no index to timeseries conversion needed (all is joined later)
    df_series = pd.DataFrame(df[df.columns[4:]].sum(),columns=["recovered"])
    return df_series
```

```python
us_con_series = confirmed_timeseries(us_confirmed)
us_dea_series = deaths_timeseries(us_deaths)
us_reco_series = reco_timeseries(us_reco)

germany_con_series = confirmed_timeseries(germany_confirmed)
germany_dea_series = deaths_timeseries(germany_deaths)
germany_reco_series = reco_timeseries(germany_reco)

italy_con_series = confirmed_timeseries(italy_confirmed)
italy_dea_series = deaths_timeseries(italy_deaths)
italy_reco_series = reco_timeseries(italy_reco)

sk_con_series = confirmed_timeseries(sk_confirmed)
sk_dea_series = deaths_timeseries(sk_deaths)
sk_reco_series = reco_timeseries(sk_reco)
```

```python
# join all data frames for each county (makes it easier to graph and compare)

us_df = us_con_series.join(us_dea_series, how = "inner")
us_df = us_df.join(us_reco_series, how = "inner")

germany_df = germany_con_series.join(germany_dea_series, how = "inner")
germany_df = germany_df.join(germany_reco_series, how = "inner")

italy_df = italy_con_series.join(italy_dea_series, how = "inner")
italy_df = italy_df.join(italy_reco_series, how = "inner")

sk_df = sk_con_series.join(sk_dea_series, how = "inner")
sk_df = sk_df.join(sk_reco_series, how = "inner")
```

```python
us_df
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
      <th>confirmed</th>
      <th>deaths</th>
      <th>recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-22</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-23</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-25</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-26</th>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-10-27</th>
      <td>8778055</td>
      <td>226696</td>
      <td>3487666</td>
    </tr>
    <tr>
      <th>2020-10-28</th>
      <td>8856413</td>
      <td>227685</td>
      <td>3518140</td>
    </tr>
    <tr>
      <th>2020-10-29</th>
      <td>8944934</td>
      <td>228656</td>
      <td>3554336</td>
    </tr>
    <tr>
      <th>2020-10-30</th>
      <td>9044255</td>
      <td>229686</td>
      <td>3578452</td>
    </tr>
    <tr>
      <th>2020-10-31</th>
      <td>9125482</td>
      <td>230548</td>
      <td>3612478</td>
    </tr>
  </tbody>
</table>
<p>284 rows × 3 columns</p>
</div>

# Visual and Descriptive Analysis

data visualization and descriptive analysis for each country

## USA

```python
us_df.plot(figsize=(14,7),title="United States confirmed, deaths and recoverd cases")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7f541b66df50>

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_18_1.png)

the number of confirmed cases started up slow until around April, it started to go up at a much faster rate and it kept that pace even during quarantine, in July the rate at which the cases are increasing got higher and the cases started increasing faster, this can be attributed to the recent protests and people's ignorance to the CDC guidelines.

deaths are the only cases that have had a continuously increasing rate, all the way from April the number of deaths is increasing at an increasing rate, until august where the increase rate is slower despite the higher number of cases.

when it comes to the recoveries, the recovery starts at the same time as the confirmed cases with a very unstable increase rate, the highest increase rate is also from around July which is surprising considering the rate of confirmed cases also went up around that time.

```python
us_cases_outcome = (us_df.tail(1)['deaths'] + us_df.tail(1)['recovered'])[0]
us_outcome_perc = (us_cases_outcome / us_df.tail(1)['confirmed'] * 100)[0]
us_death_perc = (us_df.tail(1)['deaths'] / us_cases_outcome * 100)[0]
us_reco_perc = (us_df.tail(1)['recovered'] / us_cases_outcome * 100)[0]
us_active = (us_df.tail(1)['confirmed'] - us_cases_outcome)[0]

print(f"Number of cases which had an outcome: {us_cases_outcome}")
print(f"percentage of cases that had an outcome: {round(us_outcome_perc, 2)}%")
print(f"Deaths rate: {round(us_death_perc, 2)}%")
print(f"Recovery rate: {round(us_reco_perc, 2)}%")
print(f"Currently Active cases: {us_active}")
```

    Number of cases which had an outcome: 3843026
    percentage of cases that had an outcome: 42.11%
    Deaths rate: 6.0%
    Recovery rate: 94.0%
    Currently Active cases: 5282456

the percentage of cases that had an outcome is just 38.06% of the total cases, which is very low, the other 61.4 of the cases which are not accounted for have probably not been released officially by the government, however, the recovery rate is high at 91.79% while the death rate is at 8.21%

number of currently active cases is still very high, and it's going up if the current increase rates are to be quoted.

# Modeling

for modeling and predicting the number of cases in the upcoming days the following types of models will be implemented:

- Bidrectional Long Short Term Memory (BiLSTM)
  > LSTMs' are known and widely used in time sensitive data where a variable is increaing with time depending on the values from prior days.
- Autoregressive Integrated Moving Average (ARIMA)
  > models the next step in the sequence as a linear function of the observations and resiudal errors at prior time steps.
- Holt's Exponential Smoothing (HES)
  > also referred to as holt's linear trend model or double exponential smoothing, models the next time step as an exponentially weighted linear function of observations at prior time step taking into account trends (the only difference from SES)

each country will have a total number of 3 models and the results will be compared accordingly.

our data is in a daily format and we want to predict n days at a time so we will take out the last n days and use them to test and predict outcomes it 2 weeks time.

```python
n_input = 10  # number of steps
n_features = 1 # number of y

# prepare required input data
def prepare_data(df):
    # drop rows with zeros
    df = df[(df.T != 0).any()]

    num_days = len(df) - n_input
    train = df.iloc[:num_days]
    test = df.iloc[num_days:]

    # normalize the data according to largest value
    scaler = MinMaxScaler()
    scaler.fit(train) # find max value

    scaled_train = scaler.transform(train) # divide every point by max value
    scaled_test = scaler.transform(test)

    # feed in batches [t1,t2,t3] --> t4
    generator = TimeseriesGenerator(scaled_train,scaled_train,length = n_input,batch_size = 1)
    validation_set = np.append(scaled_train[55],scaled_test) # random tbh
    validation_set = validation_set.reshape(n_input + 1,1)
    validation_gen = TimeseriesGenerator(validation_set,validation_set,length = n_input,batch_size = 1)

    return scaler, train, test, scaled_train, scaled_test, generator, validation_gen
```

## Building the models

```python
# create, train and return LSTM model
def train_lstm_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(84, recurrent_dropout = 0, unroll = False, return_sequences = True, use_bias = True, input_shape = (n_input,n_features))))
    model.add(LSTM(84, recurrent_dropout = 0.1, use_bias = True, return_sequences = True,))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(84, activation = "relu"))
    model.add(Dense(units = 1))

    # compile model
    model.compile(loss = 'mae', optimizer = Adam(1e-5))

    # finally train the model using generators
    model.fit_generator(generator,validation_data = validation_gen, epochs = 100, steps_per_epoch = round(len(train) / n_input), verbose = 0)

    return model
```

```python
# predict, rescale and append needed columns to final data frame
def lstm_predict(model):
    # holding predictions
    test_prediction = []

    # last n points from training set
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape(1,n_input,n_features)

    # predict first x days from testing data
    for i in range(len(test) + n_input):
        current_pred = model.predict(current_batch)[0]
        test_prediction.append(current_pred)
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
    # inverse scaled data
    true_prediction = scaler.inverse_transform(test_prediction)

    MAPE, accuracy, sum_errs, interval, stdev, df_forecast = gen_metrics(true_prediction)

    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast

# plotting model losses
def plot_lstm_losses(model):
    pd.DataFrame(model.history.history).plot(figsize = (14,7), title = "loss vs epochs curve")
```

```python
'''
incrementally trained ARIMA:
    - train with original train data
    - predict the next value
    - appened the prediction value to the training data
    - repeat training and appending for n times (days in this case)

    this incremental technique significantly improves the accuracy
    by always using all data up to previous day for predeicting next value
    unlike predecting multiple values at the same time which is not incremeital.

    PARAMETERS:
    p: autoregressive(AR) order
    d: order of differencing
    q: moving average(MA) order
'''

def arima_predict(p: int, d: int, q: int):
    values = [x for x in train.values]
    predictions = []
    for t in range(len(test) + n_input): # the number of testing days + the future days to predict
        model = ARIMA(values, order = (p,d,q))
        model_fit = model.fit()
        fcast = model_fit.forecast()
        predictions.append(fcast[0][0])
        values.append(fcast[0])

    MAPE, accuracy, sum_errs, interval, stdev, df_forecast = gen_metrics(predictions)

    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast
```

```python
'''
incremental Holt's (Method) Exponential Smoothing
    - trained the same way as above arima
'''
def hes_predict():
    values = [x for x in train.values]
    predictions = []
    for t in range(len(test) + n_input): # the number of testing days + the future days to predict
        model = Holt(values)
        model_fit = model.fit()
        fcast = model_fit.predict()
        predictions.append(fcast[0])
        values.append(fcast[0])

    MAPE, accuracy, sum_errs, interval, stdev, df_forecast = gen_metrics(predictions)

    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast
```

```python
# generate a dataframe with given range
def get_range_df(start: str, end: str, df):
    target_df = df.loc[pd.to_datetime(start, format='%Y-%m-%d'):pd.to_datetime(end, format='%Y-%m-%d')]
    return target_df
```

```python
# fill na values in a range predicted data frame with actual values from the original dataframe
def pad_range_df(df, original_df):
    df['confirmed'] = df.confirmed.fillna(original_df['confirmed']) # fill confirmed Na

    # fill daily na
    daily_act = []
    daily_df = pd.DataFrame(columns = ["daily"], index = df[n_input:].index)

    for num in range(n_input - 1, (n_input * 2) - 1):
        daily_act.append(df["confirmed"].iloc[num + 1] - df["confirmed"].iloc[num])

    daily_df['daily'] = daily_act
    df['daily'] = df.daily.fillna(daily_df['daily'])
    return df
```

```python
# generate metrics and final df
def gen_metrics(pred):
    # create time series
    time_series_array = test.index
    for k in range(0, n_input):
        time_series_array = time_series_array.append(time_series_array[-1:] + pd.DateOffset(1))

    # create time series data frame
    df_forecast = pd.DataFrame(columns = ["confirmed","confirmed_predicted"],index = time_series_array)

    # append confirmed and predicted confirmed
    df_forecast.loc[:,"confirmed_predicted"] = pred
    df_forecast.loc[:,"confirmed"] = test["confirmed"]

    # create and append daily cases (for both actual and predicted)
    daily_act = []
    daily_pred = []

    #actual
    daily_act.append(abs(df_forecast["confirmed"].iloc[1] - train["confirmed"].iloc[-1]))
    for num in range((n_input * 2) - 1):
        daily_act.append(df_forecast["confirmed"].iloc[num + 1] - df_forecast["confirmed"].iloc[num])

    # predicted
    daily_pred.append(df_forecast["confirmed_predicted"].iloc[1] - train["confirmed"].iloc[-1])
    for num in range((n_input * 2) - 1):
        daily_pred.append(df_forecast["confirmed_predicted"].iloc[num + 1] - df_forecast["confirmed_predicted"].iloc[num])

    df_forecast["daily"] = daily_act
    df_forecast["daily_predicted"] = daily_pred

    # calculate mean absolute percentage error
    MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:n_input]) - np.array(df_forecast["confirmed_predicted"][:n_input])) / np.array(df_forecast["confirmed"][:n_input]))

    accuracy = round((1 - MAPE) * 100, 2)

    # the error rate
    sum_errs = np.sum((np.array(df_forecast["confirmed"][:n_input]) - np.array(df_forecast["confirmed_predicted"][:n_input])) ** 2)

    # error standard deviation
    stdev = np.sqrt(1 / (n_input - 2) * sum_errs)

    # calculate prediction interval
    interval = 1.96 * stdev

    # append the min and max cases to final df
    df_forecast["confirm_min"] = df_forecast["confirmed_predicted"] - interval
    df_forecast["confirm_max"] = df_forecast["confirmed_predicted"] + interval

    # round all df values to 0 decimal points
    df_forecast = df_forecast.round()

    return MAPE, accuracy, sum_errs, interval, stdev, df_forecast
```

```python
# print metrics for given county
def print_metrics(mape, accuracy, errs, interval, std, model_type):
    m_str = "LSTM" if model_type == 0 else "ARIMA" if model_type == 1 else "HES"
    print(f"{m_str} MAPE: {round(mape * 100, 2)}%")
    print(f"{m_str} accuracy: {accuracy}%")
    print(f"{m_str} sum of errors: {round(errs)}")
    print(f"{m_str} prediction interval: {round(interval)}")
    print(f"{m_str} standard deviation: {std}")
```

```python
# for plotting the range of predicetions
def plot_results(df, country, algo):
    fig, (ax1, ax2) = plt.subplots(2, figsize = (14,20))
    ax1.set_title(f"{country} {algo} confirmed predictions")
    ax1.plot(df.index,df["confirmed"], label = "confirmed")
    ax1.plot(df.index,df["confirmed_predicted"], label = "confirmed_predicted")
    ax1.fill_between(df.index,df["confirm_min"], df["confirm_max"], color = "indigo",alpha = 0.09,label = "Confidence Interval")
    ax1.legend(loc = 2)

    ax2.set_title(f"{country} {algo} confirmed daily predictions")
    ax2.plot(df.index, df["daily"], label = "daily")
    ax2.plot(df.index, df["daily_predicted"], label = "daily_predicted")
    ax2.legend()

    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    fig.show()
```

## USA Predictions

```python
# prepare the data

scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(us_con_series)
```

```python
# train lstm model
us_lstm_model = train_lstm_model()

# plot lstm losses
plot_lstm_losses(us_lstm_model)
```

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_37_0.png)

```python
# Long short memory method
us_mape, us_accuracy, us_errs, us_interval, us_std, us_lstm_df = lstm_predict(us_lstm_model)

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 0)

us_lstm_df
```

    LSTM MAPE: 4.82%
    LSTM accuracy: 95.18%
    LSTM sum of errors: 2058053740601.0
    LSTM prediction interval: 994121.0
    LSTM standard deviation: 507204.8083122943

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
      <th>confirmed</th>
      <th>confirmed_predicted</th>
      <th>daily</th>
      <th>daily_predicted</th>
      <th>confirm_min</th>
      <th>confirm_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-10-22</th>
      <td>8409341.0</td>
      <td>8205034.0</td>
      <td>155448.0</td>
      <td>-94666.0</td>
      <td>7210912.0</td>
      <td>9199155.0</td>
    </tr>
    <tr>
      <th>2020-10-23</th>
      <td>8493088.0</td>
      <td>8242974.0</td>
      <td>83747.0</td>
      <td>37941.0</td>
      <td>7248853.0</td>
      <td>9237096.0</td>
    </tr>
    <tr>
      <th>2020-10-24</th>
      <td>8576818.0</td>
      <td>8276551.0</td>
      <td>83730.0</td>
      <td>33577.0</td>
      <td>7282430.0</td>
      <td>9270672.0</td>
    </tr>
    <tr>
      <th>2020-10-25</th>
      <td>8637625.0</td>
      <td>8306049.0</td>
      <td>60807.0</td>
      <td>29498.0</td>
      <td>7311928.0</td>
      <td>9300171.0</td>
    </tr>
    <tr>
      <th>2020-10-26</th>
      <td>8704423.0</td>
      <td>8331693.0</td>
      <td>66798.0</td>
      <td>25644.0</td>
      <td>7337572.0</td>
      <td>9325815.0</td>
    </tr>
    <tr>
      <th>2020-10-27</th>
      <td>8778055.0</td>
      <td>8354006.0</td>
      <td>73632.0</td>
      <td>22313.0</td>
      <td>7359885.0</td>
      <td>9348127.0</td>
    </tr>
    <tr>
      <th>2020-10-28</th>
      <td>8856413.0</td>
      <td>8373629.0</td>
      <td>78358.0</td>
      <td>19623.0</td>
      <td>7379508.0</td>
      <td>9367750.0</td>
    </tr>
    <tr>
      <th>2020-10-29</th>
      <td>8944934.0</td>
      <td>8392144.0</td>
      <td>88521.0</td>
      <td>18515.0</td>
      <td>7398022.0</td>
      <td>9386265.0</td>
    </tr>
    <tr>
      <th>2020-10-30</th>
      <td>9044255.0</td>
      <td>8408804.0</td>
      <td>99321.0</td>
      <td>16660.0</td>
      <td>7414683.0</td>
      <td>9402925.0</td>
    </tr>
    <tr>
      <th>2020-10-31</th>
      <td>9125482.0</td>
      <td>8423733.0</td>
      <td>81227.0</td>
      <td>14929.0</td>
      <td>7429611.0</td>
      <td>9417854.0</td>
    </tr>
    <tr>
      <th>2020-11-01</th>
      <td>NaN</td>
      <td>8435789.0</td>
      <td>NaN</td>
      <td>12056.0</td>
      <td>7441668.0</td>
      <td>9429910.0</td>
    </tr>
    <tr>
      <th>2020-11-02</th>
      <td>NaN</td>
      <td>8456020.0</td>
      <td>NaN</td>
      <td>20231.0</td>
      <td>7461899.0</td>
      <td>9450142.0</td>
    </tr>
    <tr>
      <th>2020-11-03</th>
      <td>NaN</td>
      <td>8474728.0</td>
      <td>NaN</td>
      <td>18708.0</td>
      <td>7480607.0</td>
      <td>9468849.0</td>
    </tr>
    <tr>
      <th>2020-11-04</th>
      <td>NaN</td>
      <td>8492286.0</td>
      <td>NaN</td>
      <td>17558.0</td>
      <td>7498164.0</td>
      <td>9486407.0</td>
    </tr>
    <tr>
      <th>2020-11-05</th>
      <td>NaN</td>
      <td>8509005.0</td>
      <td>NaN</td>
      <td>16720.0</td>
      <td>7514884.0</td>
      <td>9503127.0</td>
    </tr>
    <tr>
      <th>2020-11-06</th>
      <td>NaN</td>
      <td>8525149.0</td>
      <td>NaN</td>
      <td>16143.0</td>
      <td>7531027.0</td>
      <td>9519270.0</td>
    </tr>
    <tr>
      <th>2020-11-07</th>
      <td>NaN</td>
      <td>8540892.0</td>
      <td>NaN</td>
      <td>15744.0</td>
      <td>7546771.0</td>
      <td>9535014.0</td>
    </tr>
    <tr>
      <th>2020-11-08</th>
      <td>NaN</td>
      <td>8556397.0</td>
      <td>NaN</td>
      <td>15504.0</td>
      <td>7562275.0</td>
      <td>9550518.0</td>
    </tr>
    <tr>
      <th>2020-11-09</th>
      <td>NaN</td>
      <td>8571668.0</td>
      <td>NaN</td>
      <td>15272.0</td>
      <td>7577547.0</td>
      <td>9565790.0</td>
    </tr>
    <tr>
      <th>2020-11-10</th>
      <td>NaN</td>
      <td>8586776.0</td>
      <td>NaN</td>
      <td>15108.0</td>
      <td>7592655.0</td>
      <td>9580897.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
plot_results(us_lstm_df, "USA", "LSTM")
```

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_39_0.png)

```python
# Auto Regressive Integrated Moving Average

us_mape, us_accuracy, us_errs, us_interval, us_std, us_arima_df = arima_predict(8, 1, 1)

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 1)

us_arima_df
```

    ARIMA MAPE: 0.72%
    ARIMA accuracy: 99.28%
    ARIMA sum of errors: 58884624871.0
    ARIMA prediction interval: 168156.0
    ARIMA standard deviation: 85793.81160044324

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
      <th>confirmed</th>
      <th>confirmed_predicted</th>
      <th>daily</th>
      <th>daily_predicted</th>
      <th>confirm_min</th>
      <th>confirm_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-10-22</th>
      <td>8409341.0</td>
      <td>8406811.0</td>
      <td>155448.0</td>
      <td>138884.0</td>
      <td>8238655.0</td>
      <td>8574967.0</td>
    </tr>
    <tr>
      <th>2020-10-23</th>
      <td>8493088.0</td>
      <td>8476524.0</td>
      <td>83747.0</td>
      <td>69713.0</td>
      <td>8308368.0</td>
      <td>8644680.0</td>
    </tr>
    <tr>
      <th>2020-10-24</th>
      <td>8576818.0</td>
      <td>8536828.0</td>
      <td>83730.0</td>
      <td>60303.0</td>
      <td>8368672.0</td>
      <td>8704983.0</td>
    </tr>
    <tr>
      <th>2020-10-25</th>
      <td>8637625.0</td>
      <td>8593830.0</td>
      <td>60807.0</td>
      <td>57002.0</td>
      <td>8425674.0</td>
      <td>8761986.0</td>
    </tr>
    <tr>
      <th>2020-10-26</th>
      <td>8704423.0</td>
      <td>8654269.0</td>
      <td>66798.0</td>
      <td>60439.0</td>
      <td>8486113.0</td>
      <td>8822425.0</td>
    </tr>
    <tr>
      <th>2020-10-27</th>
      <td>8778055.0</td>
      <td>8716789.0</td>
      <td>73632.0</td>
      <td>62520.0</td>
      <td>8548633.0</td>
      <td>8884945.0</td>
    </tr>
    <tr>
      <th>2020-10-28</th>
      <td>8856413.0</td>
      <td>8782879.0</td>
      <td>78358.0</td>
      <td>66089.0</td>
      <td>8614723.0</td>
      <td>8951034.0</td>
    </tr>
    <tr>
      <th>2020-10-29</th>
      <td>8944934.0</td>
      <td>8853301.0</td>
      <td>88521.0</td>
      <td>70422.0</td>
      <td>8685145.0</td>
      <td>9021457.0</td>
    </tr>
    <tr>
      <th>2020-10-30</th>
      <td>9044255.0</td>
      <td>8921778.0</td>
      <td>99321.0</td>
      <td>68477.0</td>
      <td>8753622.0</td>
      <td>9089933.0</td>
    </tr>
    <tr>
      <th>2020-10-31</th>
      <td>9125482.0</td>
      <td>8984015.0</td>
      <td>81227.0</td>
      <td>62237.0</td>
      <td>8815859.0</td>
      <td>9152170.0</td>
    </tr>
    <tr>
      <th>2020-11-01</th>
      <td>NaN</td>
      <td>9043905.0</td>
      <td>NaN</td>
      <td>59891.0</td>
      <td>8875750.0</td>
      <td>9212061.0</td>
    </tr>
    <tr>
      <th>2020-11-02</th>
      <td>NaN</td>
      <td>9104850.0</td>
      <td>NaN</td>
      <td>60944.0</td>
      <td>8936694.0</td>
      <td>9273006.0</td>
    </tr>
    <tr>
      <th>2020-11-03</th>
      <td>NaN</td>
      <td>9167791.0</td>
      <td>NaN</td>
      <td>62941.0</td>
      <td>8999635.0</td>
      <td>9335947.0</td>
    </tr>
    <tr>
      <th>2020-11-04</th>
      <td>NaN</td>
      <td>9234322.0</td>
      <td>NaN</td>
      <td>66531.0</td>
      <td>9066166.0</td>
      <td>9402478.0</td>
    </tr>
    <tr>
      <th>2020-11-05</th>
      <td>NaN</td>
      <td>9303277.0</td>
      <td>NaN</td>
      <td>68955.0</td>
      <td>9135121.0</td>
      <td>9471433.0</td>
    </tr>
    <tr>
      <th>2020-11-06</th>
      <td>NaN</td>
      <td>9369836.0</td>
      <td>NaN</td>
      <td>66560.0</td>
      <td>9201681.0</td>
      <td>9537992.0</td>
    </tr>
    <tr>
      <th>2020-11-07</th>
      <td>NaN</td>
      <td>9431964.0</td>
      <td>NaN</td>
      <td>62128.0</td>
      <td>9263808.0</td>
      <td>9600120.0</td>
    </tr>
    <tr>
      <th>2020-11-08</th>
      <td>NaN</td>
      <td>9491890.0</td>
      <td>NaN</td>
      <td>59926.0</td>
      <td>9323734.0</td>
      <td>9660046.0</td>
    </tr>
    <tr>
      <th>2020-11-09</th>
      <td>NaN</td>
      <td>9551959.0</td>
      <td>NaN</td>
      <td>60069.0</td>
      <td>9383803.0</td>
      <td>9720114.0</td>
    </tr>
    <tr>
      <th>2020-11-10</th>
      <td>NaN</td>
      <td>9613919.0</td>
      <td>NaN</td>
      <td>61961.0</td>
      <td>9445763.0</td>
      <td>9782075.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
plot_results(us_arima_df, "USA", "incremental ARIMA")
```

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_41_0.png)

```python
# Holts Exponential Smoothing
us_mape, us_accuracy, us_errs, us_interval, us_std, us_hes_df = hes_predict()

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 2)

us_hes_df
```

    HES MAPE: 0.83%
    HES accuracy: 99.17%
    HES sum of errors: 75839516016.0
    HES prediction interval: 190835.0
    HES standard deviation: 97364.98088117794

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
      <th>confirmed</th>
      <th>confirmed_predicted</th>
      <th>daily</th>
      <th>daily_predicted</th>
      <th>confirm_min</th>
      <th>confirm_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-10-22</th>
      <td>8409341.0</td>
      <td>8400415.0</td>
      <td>155448.0</td>
      <td>125551.0</td>
      <td>8209580.0</td>
      <td>8591251.0</td>
    </tr>
    <tr>
      <th>2020-10-23</th>
      <td>8493088.0</td>
      <td>8463191.0</td>
      <td>83747.0</td>
      <td>62776.0</td>
      <td>8272356.0</td>
      <td>8654027.0</td>
    </tr>
    <tr>
      <th>2020-10-24</th>
      <td>8576818.0</td>
      <td>8525967.0</td>
      <td>83730.0</td>
      <td>62776.0</td>
      <td>8335132.0</td>
      <td>8716802.0</td>
    </tr>
    <tr>
      <th>2020-10-25</th>
      <td>8637625.0</td>
      <td>8588743.0</td>
      <td>60807.0</td>
      <td>62776.0</td>
      <td>8397907.0</td>
      <td>8779578.0</td>
    </tr>
    <tr>
      <th>2020-10-26</th>
      <td>8704423.0</td>
      <td>8651518.0</td>
      <td>66798.0</td>
      <td>62776.0</td>
      <td>8460683.0</td>
      <td>8842354.0</td>
    </tr>
    <tr>
      <th>2020-10-27</th>
      <td>8778055.0</td>
      <td>8714294.0</td>
      <td>73632.0</td>
      <td>62776.0</td>
      <td>8523459.0</td>
      <td>8905129.0</td>
    </tr>
    <tr>
      <th>2020-10-28</th>
      <td>8856413.0</td>
      <td>8777070.0</td>
      <td>78358.0</td>
      <td>62776.0</td>
      <td>8586234.0</td>
      <td>8967905.0</td>
    </tr>
    <tr>
      <th>2020-10-29</th>
      <td>8944934.0</td>
      <td>8839845.0</td>
      <td>88521.0</td>
      <td>62776.0</td>
      <td>8649010.0</td>
      <td>9030681.0</td>
    </tr>
    <tr>
      <th>2020-10-30</th>
      <td>9044255.0</td>
      <td>8902621.0</td>
      <td>99321.0</td>
      <td>62776.0</td>
      <td>8711786.0</td>
      <td>9093457.0</td>
    </tr>
    <tr>
      <th>2020-10-31</th>
      <td>9125482.0</td>
      <td>8965397.0</td>
      <td>81227.0</td>
      <td>62776.0</td>
      <td>8774562.0</td>
      <td>9156232.0</td>
    </tr>
    <tr>
      <th>2020-11-01</th>
      <td>NaN</td>
      <td>9028173.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>8837337.0</td>
      <td>9219008.0</td>
    </tr>
    <tr>
      <th>2020-11-02</th>
      <td>NaN</td>
      <td>9090948.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>8900113.0</td>
      <td>9281784.0</td>
    </tr>
    <tr>
      <th>2020-11-03</th>
      <td>NaN</td>
      <td>9153724.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>8962889.0</td>
      <td>9344559.0</td>
    </tr>
    <tr>
      <th>2020-11-04</th>
      <td>NaN</td>
      <td>9216500.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9025664.0</td>
      <td>9407335.0</td>
    </tr>
    <tr>
      <th>2020-11-05</th>
      <td>NaN</td>
      <td>9279275.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9088440.0</td>
      <td>9470111.0</td>
    </tr>
    <tr>
      <th>2020-11-06</th>
      <td>NaN</td>
      <td>9342051.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9151216.0</td>
      <td>9532887.0</td>
    </tr>
    <tr>
      <th>2020-11-07</th>
      <td>NaN</td>
      <td>9404827.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9213992.0</td>
      <td>9595662.0</td>
    </tr>
    <tr>
      <th>2020-11-08</th>
      <td>NaN</td>
      <td>9467603.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9276767.0</td>
      <td>9658438.0</td>
    </tr>
    <tr>
      <th>2020-11-09</th>
      <td>NaN</td>
      <td>9530378.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9339543.0</td>
      <td>9721214.0</td>
    </tr>
    <tr>
      <th>2020-11-10</th>
      <td>NaN</td>
      <td>9593154.0</td>
      <td>NaN</td>
      <td>62776.0</td>
      <td>9402319.0</td>
      <td>9783989.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
plot_results(us_hes_df, "USA", "incremental HES")
```

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_43_0.png)

# Effectiveness of mandated lockdown

was the US lockdown effective in reducing the cases?  
the US started their lockdwon in 2020-03-17 and it was ended by the erupting protests.
tracking the lockdown might be tricky in the US at least because each state started their lockdown at their own pace and there was no federally mandated lockdown while some other states never went into lockdowns, taking that into account we will consider the end of the lockdown to be the end of may which was the start of the Gorge Floyed protests.

**Time frame**
from 2020-03-17 until 2020-05-31

```python
us_lockdown = get_range_df('2020-03-17', '2020-05-31', us_con_series)
```

```python
fig, ax = plt.subplots(1, figsize = (14,7))
ax.plot(us_lockdown.index, us_lockdown['confirmed'], label = 'confirmed')
ax.plot(us_lockdown.index, us_lockdown.rolling(7).mean(), label = 'confirmed mean')
ax.legend()
```

    <matplotlib.legend.Legend at 0x7f53d0630f90>

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_46_1.png)

the actual values are above the moving average of each 7 days meaning the lockdown did not work as inteded and the number of cases was still very high when compared to the average of each 7 days, to make sure our previous model predictions are accurate we will use this period of time as a base and train the model on it and do prediction for the days after that which we already have the data on.
we will use the ARIMA model becuase the amount of data we have will not train a neural network ideally.

```python
scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(us_lockdown)
```

```python
# Auto Regressive Integrated Moving Average

us_mape, us_accuracy, us_errs, us_interval, us_std, us_arima_df = arima_predict(8, 1, 1)

print_metrics(us_mape, us_accuracy, us_errs, us_interval, us_std, 1)

us_arima_df = pad_range_df(us_arima_df, us_con_series)

us_arima_df
```

    ARIMA MAPE: 0.55%
    ARIMA accuracy: 99.45%
    ARIMA sum of errors: 1187672403.0
    ARIMA prediction interval: 23881.0
    ARIMA standard deviation: 12184.377305753513

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
      <th>confirmed</th>
      <th>confirmed_predicted</th>
      <th>daily</th>
      <th>daily_predicted</th>
      <th>confirm_min</th>
      <th>confirm_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-22</th>
      <td>1608604.0</td>
      <td>1610272.0</td>
      <td>44636.0</td>
      <td>47633.0</td>
      <td>1586391.0</td>
      <td>1634154.0</td>
    </tr>
    <tr>
      <th>2020-05-23</th>
      <td>1629802.0</td>
      <td>1632799.0</td>
      <td>21198.0</td>
      <td>22527.0</td>
      <td>1608918.0</td>
      <td>1656681.0</td>
    </tr>
    <tr>
      <th>2020-05-24</th>
      <td>1649916.0</td>
      <td>1653311.0</td>
      <td>20114.0</td>
      <td>20511.0</td>
      <td>1629429.0</td>
      <td>1677192.0</td>
    </tr>
    <tr>
      <th>2020-05-25</th>
      <td>1668235.0</td>
      <td>1674419.0</td>
      <td>18319.0</td>
      <td>21108.0</td>
      <td>1650537.0</td>
      <td>1698300.0</td>
    </tr>
    <tr>
      <th>2020-05-26</th>
      <td>1687761.0</td>
      <td>1695924.0</td>
      <td>19526.0</td>
      <td>21505.0</td>
      <td>1672042.0</td>
      <td>1719805.0</td>
    </tr>
    <tr>
      <th>2020-05-27</th>
      <td>1706351.0</td>
      <td>1719538.0</td>
      <td>18590.0</td>
      <td>23614.0</td>
      <td>1695657.0</td>
      <td>1743419.0</td>
    </tr>
    <tr>
      <th>2020-05-28</th>
      <td>1729299.0</td>
      <td>1744487.0</td>
      <td>22948.0</td>
      <td>24949.0</td>
      <td>1720606.0</td>
      <td>1768369.0</td>
    </tr>
    <tr>
      <th>2020-05-29</th>
      <td>1753651.0</td>
      <td>1768770.0</td>
      <td>24352.0</td>
      <td>24283.0</td>
      <td>1744889.0</td>
      <td>1792652.0</td>
    </tr>
    <tr>
      <th>2020-05-30</th>
      <td>1777495.0</td>
      <td>1791120.0</td>
      <td>23844.0</td>
      <td>22350.0</td>
      <td>1767239.0</td>
      <td>1815002.0</td>
    </tr>
    <tr>
      <th>2020-05-31</th>
      <td>1796670.0</td>
      <td>1812184.0</td>
      <td>19175.0</td>
      <td>21064.0</td>
      <td>1788303.0</td>
      <td>1836066.0</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>1814034.0</td>
      <td>1833132.0</td>
      <td>17364.0</td>
      <td>20948.0</td>
      <td>1809251.0</td>
      <td>1857014.0</td>
    </tr>
    <tr>
      <th>2020-06-02</th>
      <td>1835408.0</td>
      <td>1854887.0</td>
      <td>21374.0</td>
      <td>21755.0</td>
      <td>1831006.0</td>
      <td>1878768.0</td>
    </tr>
    <tr>
      <th>2020-06-03</th>
      <td>1855386.0</td>
      <td>1878310.0</td>
      <td>19978.0</td>
      <td>23423.0</td>
      <td>1854428.0</td>
      <td>1902191.0</td>
    </tr>
    <tr>
      <th>2020-06-04</th>
      <td>1877125.0</td>
      <td>1902606.0</td>
      <td>21739.0</td>
      <td>24296.0</td>
      <td>1878724.0</td>
      <td>1926487.0</td>
    </tr>
    <tr>
      <th>2020-06-05</th>
      <td>1902294.0</td>
      <td>1926287.0</td>
      <td>25169.0</td>
      <td>23682.0</td>
      <td>1902406.0</td>
      <td>1950169.0</td>
    </tr>
    <tr>
      <th>2020-06-06</th>
      <td>1924132.0</td>
      <td>1948548.0</td>
      <td>21838.0</td>
      <td>22261.0</td>
      <td>1924667.0</td>
      <td>1972430.0</td>
    </tr>
    <tr>
      <th>2020-06-07</th>
      <td>1941920.0</td>
      <td>1969746.0</td>
      <td>17788.0</td>
      <td>21198.0</td>
      <td>1945864.0</td>
      <td>1993627.0</td>
    </tr>
    <tr>
      <th>2020-06-08</th>
      <td>1959448.0</td>
      <td>1990779.0</td>
      <td>17528.0</td>
      <td>21033.0</td>
      <td>1966898.0</td>
      <td>2014661.0</td>
    </tr>
    <tr>
      <th>2020-06-09</th>
      <td>1977820.0</td>
      <td>2012666.0</td>
      <td>18372.0</td>
      <td>21887.0</td>
      <td>1988784.0</td>
      <td>2036547.0</td>
    </tr>
    <tr>
      <th>2020-06-10</th>
      <td>1998646.0</td>
      <td>2035860.0</td>
      <td>20826.0</td>
      <td>23194.0</td>
      <td>2011979.0</td>
      <td>2059742.0</td>
    </tr>
  </tbody>
</table>
</div>

```python
plot_results(us_arima_df, "USA", "incremental ARIMA")
```

![png](/assets/images/covid-19-forecasting-lstms-and-statistical-models_files/covid-19-forecasting-lstms-and-statistical-models_50_0.png)

we can see that the predicted totals and predicted daily cases are fairly accurate thus our previous predictions can be taken with some degree of accuracy, and might be used for making decisions.

# Conclusion

from all the graphs, functions and numbers above we can come to a simple conclusion that is, there is no single model that will perform best in all scenario even when the data is very similar (in trend not numbers), each model was best for a specific country and wasn’t so far behind in the others for example the HES model is the most accurate with the South Korean dataset but is almost the same as the ARIMA model in Italy.

**whats the difference between ARIMA and HES?**  
ARIMA uses a non-linear function for coefficient calculations, that’s why the graph does curve sometimes (Italy) while HES is a pure linear method that uses a linear function and is always a straight line

**Considering LSTM is usually the least accurate, is it worth the training time?**  
here may be, however, deep learning has its place among machine learning algorithms and can perform tasks these other functions could never, also the LSTM model always predicts a wider interval compared to the other 2, in a practical scenario where range is important the other 2 models will not be ideal because their results are limited by the original value and don’t spread as much, the LSTM model could provide better estimates.

**ARIMA or HES?**  
HES, because it takes much less time to train and is as accurate or even more accurate sometimes.
