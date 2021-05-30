---
title: Covid 19 Forecasting Malaysia Data Analysis
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/covid-19-forecasting-malaysia-data-analysis
tags:
  - data-science
  - machine-learning
  - project
---

this notebook attempts to answer various data science question from 4 categories

based on the COVID-19 daily cases, deaths, and recoveries, the focus is
primarily on Malaysia.

## The Questions

- Descriptive
  - when did the second wave start?
- Diagnostic
  - what is the percentage of cases that led to death (death rate)?
- Predictive
  - how many cases will Malaysia have by the end of 2020?
- Prescriptive
  - how can we make the number of cases go down before the end of 2020?

### other details

- the dataset is obtained from the center for system science and engineering
  (CSSE) https://github.com/CSSEGISandData/COVID-19

- for predictive analysis, a LSTM model is used to predict the upcoming cases

  based on solely the previous day’s data considering no other factors,

  still, LSTM are very popular for such task because of their accuracy and
  ability to generalize data.

- the used framework for the predictive model is TensorFlow 2.x

- this notebook takes some code from a larger project i have been working on for
  a while, it’s a modular architecture that can predict cases for any country
  just given a timeseries of its previous cases using 3 different models, the
  project can be found here
  https://www.kaggle.com/abubakaryagob/covid-19-forecasting-automated-edition

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation, GlobalMaxPooling1D
from keras.optimizers import Adam

%matplotlib inline
```

read in the datasets

```python
df_confirmed = pd.read_csv("../input/covid-19/time_series_covid19_confirmed_global.csv")
df_deaths = pd.read_csv("../input/covid-19/time_series_covid19_deaths_global.csv")
df_reco = pd.read_csv("../input/covid-19/time_series_covid19_recovered_global.csv")
```

after reading in our dataset lets take a look at it by showing the first few
countries for confirmed case, deaths, and recoveries

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
      <th>12/2/20</th>
      <th>12/3/20</th>
      <th>12/4/20</th>
      <th>12/5/20</th>
      <th>12/6/20</th>
      <th>12/7/20</th>
      <th>12/8/20</th>
      <th>12/9/20</th>
      <th>12/10/20</th>
      <th>12/11/20</th>
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
      <td>46718</td>
      <td>46837</td>
      <td>46837</td>
      <td>47072</td>
      <td>47306</td>
      <td>47516</td>
      <td>47716</td>
      <td>47851</td>
      <td>48053</td>
      <td>48116</td>
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
      <td>39719</td>
      <td>40501</td>
      <td>41302</td>
      <td>42148</td>
      <td>42988</td>
      <td>43683</td>
      <td>44436</td>
      <td>45188</td>
      <td>46061</td>
      <td>46863</td>
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
      <td>85084</td>
      <td>85927</td>
      <td>86730</td>
      <td>87502</td>
      <td>88252</td>
      <td>88825</td>
      <td>89416</td>
      <td>90014</td>
      <td>90579</td>
      <td>91121</td>
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
      <td>6842</td>
      <td>6904</td>
      <td>6955</td>
      <td>7005</td>
      <td>7050</td>
      <td>7084</td>
      <td>7127</td>
      <td>7162</td>
      <td>7190</td>
      <td>7236</td>
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
      <td>15319</td>
      <td>15361</td>
      <td>15493</td>
      <td>15536</td>
      <td>15591</td>
      <td>15648</td>
      <td>15729</td>
      <td>15804</td>
      <td>15925</td>
      <td>16061</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 329 columns</p>
</div>

each of the dataframes contains data for all countries, we only need malaysia so
lets extract that from it

```python
my_confirmed = df_confirmed[df_confirmed["Country/Region"] == "Malaysia"]
my_deaths = df_deaths[df_deaths["Country/Region"] == "Malaysia"]
my_reco = df_reco[df_reco["Country/Region"] == "Malaysia"]
```

now lets take a look at our data for Malaysia

```python
my_confirmed
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
      <th>12/2/20</th>
      <th>12/3/20</th>
      <th>12/4/20</th>
      <th>12/5/20</th>
      <th>12/6/20</th>
      <th>12/7/20</th>
      <th>12/8/20</th>
      <th>12/9/20</th>
      <th>12/10/20</th>
      <th>12/11/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>173</th>
      <td>NaN</td>
      <td>Malaysia</td>
      <td>4.210484</td>
      <td>101.975766</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>68020</td>
      <td>69095</td>
      <td>70236</td>
      <td>71359</td>
      <td>72694</td>
      <td>74294</td>
      <td>75306</td>
      <td>76265</td>
      <td>78499</td>
      <td>80309</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 329 columns</p>
</div>

our data is in a format that will not allow us to use it for graphing or
predictions, first we should reshape the data to a format that is more friendly
to our goal

```python
# convert passed dataframe to a timeseries (a format easy to graph and use for training models)
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
# convert each dataframe to a timeseries
my_con_series = confirmed_timeseries(my_confirmed)
my_dea_series = deaths_timeseries(my_deaths)
my_reco_series = reco_timeseries(my_reco)
```

```python
my_con_series
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-22</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-23</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-25</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2020-01-26</th>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-12-07</th>
      <td>74294</td>
    </tr>
    <tr>
      <th>2020-12-08</th>
      <td>75306</td>
    </tr>
    <tr>
      <th>2020-12-09</th>
      <td>76265</td>
    </tr>
    <tr>
      <th>2020-12-10</th>
      <td>78499</td>
    </tr>
    <tr>
      <th>2020-12-11</th>
      <td>80309</td>
    </tr>
  </tbody>
</table>
<p>325 rows × 1 columns</p>
</div>

now that we have our data as a time series, lets join all the differnet cases
(confirmed, deaths, recovred) together so its easier to graph them

```python
# join all 3 data frames
my_df = my_con_series.join(my_dea_series, how = "inner")
my_df = my_df.join(my_reco_series, how = "inner")

```

now lets take a look at our final dataframe

```python
my_df
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
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-23</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-24</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-25</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2020-01-26</th>
      <td>4</td>
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
      <th>2020-12-07</th>
      <td>74294</td>
      <td>384</td>
      <td>62306</td>
    </tr>
    <tr>
      <th>2020-12-08</th>
      <td>75306</td>
      <td>388</td>
      <td>64056</td>
    </tr>
    <tr>
      <th>2020-12-09</th>
      <td>76265</td>
      <td>393</td>
      <td>65124</td>
    </tr>
    <tr>
      <th>2020-12-10</th>
      <td>78499</td>
      <td>396</td>
      <td>66236</td>
    </tr>
    <tr>
      <th>2020-12-11</th>
      <td>80309</td>
      <td>402</td>
      <td>67173</td>
    </tr>
  </tbody>
</table>
<p>325 rows × 3 columns</p>
</div>

we can see that its now in a format that is easy to interpret, graph and use for
predictions with all columns included

# Descriptive and Diagnostic Analysis

```python
my_df.plot(figsize=(14,7),title="Malysia confirmed, deaths, and recoverd cases")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7f2ab8657a90>

![png](/assets/images/covid-19-forecasting-malaysia-data-analysis_files/covid-19-forecasting-malaysia-data-analysis_21_1.png)

### Question 1: when did the second wave start?

from the above graph we can make several remarks, one of which is that the
second wave started at the beginning of September and its still going as the
time of writing this (13-12-2020)

to calculate the percentage of cases that led to death, we first need to know
the number of cases that led to an outcome, from that we can easily extract the
number of cases that led to death and from that we can calculate the percentage
against all the cases that had an outcome

```python
my_cases_outcome = (my_df.tail(1)['deaths'] + my_df.tail(1)['recovered'])[0]
my_outcome_perc = (my_cases_outcome / my_df.tail(1)['confirmed'] * 100)[0]
my_death_perc = (my_df.tail(1)['deaths'] / my_cases_outcome * 100)[0]
my_reco_perc = (my_df.tail(1)['recovered'] / my_cases_outcome * 100)[0]
my_active = (my_df.tail(1)['confirmed'] - my_cases_outcome)[0]

print(f"Number of cases which had an outcome: {my_cases_outcome}")
print(f"percentage of cases that had an outcome: {round(my_outcome_perc, 2)}%")
print(f"Deaths rate: {round(my_death_perc, 2)}%")
print(f"Recovery rate: {round(my_reco_perc, 2)}%")
print(f"Currently Active cases: {my_active}")
```

    Number of cases which had an outcome: 67575
    percentage of cases that had an outcome: 84.14%
    Deaths rate: 0.59%
    Recovery rate: 99.41%
    Currently Active cases: 12734

### Question 2: what is the percentage of cases that led to death (death rate)?

we can see from the above results that the percentage of cases that led to death
is 0.59% which is very miniscule in the grand scheme of things, this tells us
that although Malaysia has had a considerable number of cases most of them did
not end in deaths, in simpler terms, for every 200 cases only 1 death occured.

# Predective and Prescriptive Analaysis

this is where we train an LSTM RNN to predict upcoming cases, we will make
predictions for the next 20 days, for testing the accuracy of our model, we will
take the last 20 days from the dataset and use them for testing

```python
n_input = 20  # number of steps (days to predict)
n_features = 1 # number of y (model outputs)

# preporcess a dataframe and create required vairables for training the LSTM
# train: the data used to make the training generator
# test: the data used to make the test generator
# scaler: data scaler to normalize the data (easier for the model)
# scaled_train: train dataset scaled down to the largest value in the dataset
# scaled test: train dataset scaled down to the largest value in the dataset
# generator: train data generator, used to train the model by feeding it batches of the train data
# validation_generator: validation data generator, used to validate the model during training
#                       by feeding it batches of a randomly selected points from the train dataset

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

```python
# create, train and return LSTM model
def train_lstm_model():
    model = Sequential()
    model.add(LSTM(84, recurrent_dropout = 0, return_sequences = True, input_shape = (n_input,n_features)))
    model.add(LSTM(84, recurrent_dropout = 0.1, use_bias = True, return_sequences = True,))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(84, activation = "relu"))
    model.add(Dense(1))

    # compile model
    model.compile(loss = 'mse', optimizer = Adam(1e-5))

    # finally train the model using generators
    model.fit_generator(generator,
                        validation_data = validation_gen,
                        epochs = 300,
                        steps_per_epoch = round(len(train) / n_input),
                        verbose = 0
                       )

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


    # calculate mean absolute percentage error
    MAPE = np.mean(np.abs(np.array(df_forecast["confirmed"][:n_input]) -
                          np.array(df_forecast["confirmed_predicted"][:n_input])) /
                   np.array(df_forecast["confirmed"][:n_input]))

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

    # appened daily data
    df_forecast["daily"] = daily_act
    df_forecast["daily_predicted"] = daily_pred

    daily_err = np.sum((np.array(df_forecast["daily"][:n_input]) - np.array(df_forecast["daily_predicted"][:n_input])) ** 2)
    daily_stdev = np.sqrt(1 / (n_input - 2) * daily_err)
    daily_interval = 1.96 * daily_stdev

    df_forecast["daily_min"] = df_forecast["daily_predicted"] - daily_interval
    df_forecast["daily_max"] = df_forecast["daily_predicted"] + daily_interval

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

## Train the model

```python
# prepare the data (using confirmed cases dataset)

scaler, train, test, scaled_train, scaled_test, generator, validation_gen = prepare_data(my_con_series)
```

```python
# train lstm model
my_lstm_model = train_lstm_model()

# plot lstm losses
plot_lstm_losses(my_lstm_model)
```

![png](/assets/images/covid-19-forecasting-malaysia-data-analysis_files/covid-19-forecasting-malaysia-data-analysis_35_0.png)

We can see that the number of losses is tiny at the end of our training. This
tells us the model has successfully learned the data and can - to some degree of
accuracy - make predictions based on it.

lets first calculate the accuracy for our testing dataset

```python
my_mape, my_accuracy, my_errs, my_interval, my_std, my_lstm_df = lstm_predict(my_lstm_model)

print_metrics(my_mape, my_accuracy, my_errs, my_interval, my_std, 0) # 0 here means LSTM

my_lstm_df
```

    LSTM MAPE: 2.01%
    LSTM accuracy: 97.99%
    LSTM sum of errors: 43483777.0
    LSTM prediction interval: 3046.0
    LSTM standard deviation: 1554.2732613061567

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
      <th>confirm_min</th>
      <th>confirm_max</th>
      <th>daily</th>
      <th>daily_predicted</th>
      <th>daily_min</th>
      <th>daily_max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-11-22</th>
      <td>54775.0</td>
      <td>57453.0</td>
      <td>54407.0</td>
      <td>60499.0</td>
      <td>2980.0</td>
      <td>5020.0</td>
      <td>3592.0</td>
      <td>6448.0</td>
    </tr>
    <tr>
      <th>2020-11-23</th>
      <td>56659.0</td>
      <td>58699.0</td>
      <td>55653.0</td>
      <td>61746.0</td>
      <td>1884.0</td>
      <td>1246.0</td>
      <td>-182.0</td>
      <td>2674.0</td>
    </tr>
    <tr>
      <th>2020-11-24</th>
      <td>58847.0</td>
      <td>59943.0</td>
      <td>56896.0</td>
      <td>62989.0</td>
      <td>2188.0</td>
      <td>1243.0</td>
      <td>-184.0</td>
      <td>2671.0</td>
    </tr>
    <tr>
      <th>2020-11-25</th>
      <td>59817.0</td>
      <td>61181.0</td>
      <td>58134.0</td>
      <td>64227.0</td>
      <td>970.0</td>
      <td>1238.0</td>
      <td>-190.0</td>
      <td>2666.0</td>
    </tr>
    <tr>
      <th>2020-11-26</th>
      <td>60752.0</td>
      <td>62414.0</td>
      <td>59368.0</td>
      <td>65460.0</td>
      <td>935.0</td>
      <td>1233.0</td>
      <td>-194.0</td>
      <td>2661.0</td>
    </tr>
    <tr>
      <th>2020-11-27</th>
      <td>61861.0</td>
      <td>63595.0</td>
      <td>60549.0</td>
      <td>66641.0</td>
      <td>1109.0</td>
      <td>1181.0</td>
      <td>-247.0</td>
      <td>2609.0</td>
    </tr>
    <tr>
      <th>2020-11-28</th>
      <td>63176.0</td>
      <td>64755.0</td>
      <td>61708.0</td>
      <td>67801.0</td>
      <td>1315.0</td>
      <td>1160.0</td>
      <td>-268.0</td>
      <td>2588.0</td>
    </tr>
    <tr>
      <th>2020-11-29</th>
      <td>64485.0</td>
      <td>65907.0</td>
      <td>62861.0</td>
      <td>68954.0</td>
      <td>1309.0</td>
      <td>1152.0</td>
      <td>-275.0</td>
      <td>2580.0</td>
    </tr>
    <tr>
      <th>2020-11-30</th>
      <td>65697.0</td>
      <td>67042.0</td>
      <td>63995.0</td>
      <td>70088.0</td>
      <td>1212.0</td>
      <td>1134.0</td>
      <td>-293.0</td>
      <td>2562.0</td>
    </tr>
    <tr>
      <th>2020-12-01</th>
      <td>67169.0</td>
      <td>68159.0</td>
      <td>65113.0</td>
      <td>71205.0</td>
      <td>1472.0</td>
      <td>1117.0</td>
      <td>-310.0</td>
      <td>2545.0</td>
    </tr>
    <tr>
      <th>2020-12-02</th>
      <td>68020.0</td>
      <td>69259.0</td>
      <td>66212.0</td>
      <td>72305.0</td>
      <td>851.0</td>
      <td>1100.0</td>
      <td>-328.0</td>
      <td>2528.0</td>
    </tr>
    <tr>
      <th>2020-12-03</th>
      <td>69095.0</td>
      <td>70334.0</td>
      <td>67288.0</td>
      <td>73381.0</td>
      <td>1075.0</td>
      <td>1076.0</td>
      <td>-352.0</td>
      <td>2503.0</td>
    </tr>
    <tr>
      <th>2020-12-04</th>
      <td>70236.0</td>
      <td>71364.0</td>
      <td>68317.0</td>
      <td>74410.0</td>
      <td>1141.0</td>
      <td>1030.0</td>
      <td>-398.0</td>
      <td>2457.0</td>
    </tr>
    <tr>
      <th>2020-12-05</th>
      <td>71359.0</td>
      <td>72356.0</td>
      <td>69310.0</td>
      <td>75403.0</td>
      <td>1123.0</td>
      <td>993.0</td>
      <td>-435.0</td>
      <td>2420.0</td>
    </tr>
    <tr>
      <th>2020-12-06</th>
      <td>72694.0</td>
      <td>73306.0</td>
      <td>70260.0</td>
      <td>76352.0</td>
      <td>1335.0</td>
      <td>950.0</td>
      <td>-478.0</td>
      <td>2377.0</td>
    </tr>
    <tr>
      <th>2020-12-07</th>
      <td>74294.0</td>
      <td>74218.0</td>
      <td>71172.0</td>
      <td>77265.0</td>
      <td>1600.0</td>
      <td>912.0</td>
      <td>-515.0</td>
      <td>2340.0</td>
    </tr>
    <tr>
      <th>2020-12-08</th>
      <td>75306.0</td>
      <td>75105.0</td>
      <td>72059.0</td>
      <td>78151.0</td>
      <td>1012.0</td>
      <td>887.0</td>
      <td>-541.0</td>
      <td>2314.0</td>
    </tr>
    <tr>
      <th>2020-12-09</th>
      <td>76265.0</td>
      <td>75981.0</td>
      <td>72934.0</td>
      <td>79027.0</td>
      <td>959.0</td>
      <td>876.0</td>
      <td>-552.0</td>
      <td>2304.0</td>
    </tr>
    <tr>
      <th>2020-12-10</th>
      <td>78499.0</td>
      <td>76808.0</td>
      <td>73762.0</td>
      <td>79855.0</td>
      <td>2234.0</td>
      <td>828.0</td>
      <td>-600.0</td>
      <td>2255.0</td>
    </tr>
    <tr>
      <th>2020-12-11</th>
      <td>80309.0</td>
      <td>77603.0</td>
      <td>74557.0</td>
      <td>80650.0</td>
      <td>1810.0</td>
      <td>795.0</td>
      <td>-633.0</td>
      <td>2223.0</td>
    </tr>
    <tr>
      <th>2020-12-12</th>
      <td>NaN</td>
      <td>78376.0</td>
      <td>75330.0</td>
      <td>81423.0</td>
      <td>NaN</td>
      <td>773.0</td>
      <td>-655.0</td>
      <td>2201.0</td>
    </tr>
    <tr>
      <th>2020-12-13</th>
      <td>NaN</td>
      <td>79020.0</td>
      <td>75974.0</td>
      <td>82066.0</td>
      <td>NaN</td>
      <td>644.0</td>
      <td>-784.0</td>
      <td>2072.0</td>
    </tr>
    <tr>
      <th>2020-12-14</th>
      <td>NaN</td>
      <td>79629.0</td>
      <td>76583.0</td>
      <td>82676.0</td>
      <td>NaN</td>
      <td>609.0</td>
      <td>-818.0</td>
      <td>2037.0</td>
    </tr>
    <tr>
      <th>2020-12-15</th>
      <td>NaN</td>
      <td>80205.0</td>
      <td>77159.0</td>
      <td>83251.0</td>
      <td>NaN</td>
      <td>576.0</td>
      <td>-852.0</td>
      <td>2003.0</td>
    </tr>
    <tr>
      <th>2020-12-16</th>
      <td>NaN</td>
      <td>80748.0</td>
      <td>77701.0</td>
      <td>83794.0</td>
      <td>NaN</td>
      <td>543.0</td>
      <td>-885.0</td>
      <td>1970.0</td>
    </tr>
    <tr>
      <th>2020-12-17</th>
      <td>NaN</td>
      <td>81258.0</td>
      <td>78211.0</td>
      <td>84304.0</td>
      <td>NaN</td>
      <td>510.0</td>
      <td>-918.0</td>
      <td>1938.0</td>
    </tr>
    <tr>
      <th>2020-12-18</th>
      <td>NaN</td>
      <td>81738.0</td>
      <td>78691.0</td>
      <td>84784.0</td>
      <td>NaN</td>
      <td>480.0</td>
      <td>-948.0</td>
      <td>1908.0</td>
    </tr>
    <tr>
      <th>2020-12-19</th>
      <td>NaN</td>
      <td>82189.0</td>
      <td>79142.0</td>
      <td>85235.0</td>
      <td>NaN</td>
      <td>451.0</td>
      <td>-977.0</td>
      <td>1879.0</td>
    </tr>
    <tr>
      <th>2020-12-20</th>
      <td>NaN</td>
      <td>82611.0</td>
      <td>79565.0</td>
      <td>85658.0</td>
      <td>NaN</td>
      <td>423.0</td>
      <td>-1005.0</td>
      <td>1850.0</td>
    </tr>
    <tr>
      <th>2020-12-21</th>
      <td>NaN</td>
      <td>83014.0</td>
      <td>79968.0</td>
      <td>86061.0</td>
      <td>NaN</td>
      <td>403.0</td>
      <td>-1025.0</td>
      <td>1831.0</td>
    </tr>
    <tr>
      <th>2020-12-22</th>
      <td>NaN</td>
      <td>83397.0</td>
      <td>80350.0</td>
      <td>86443.0</td>
      <td>NaN</td>
      <td>382.0</td>
      <td>-1045.0</td>
      <td>1810.0</td>
    </tr>
    <tr>
      <th>2020-12-23</th>
      <td>NaN</td>
      <td>83753.0</td>
      <td>80707.0</td>
      <td>86799.0</td>
      <td>NaN</td>
      <td>356.0</td>
      <td>-1071.0</td>
      <td>1784.0</td>
    </tr>
    <tr>
      <th>2020-12-24</th>
      <td>NaN</td>
      <td>84084.0</td>
      <td>81038.0</td>
      <td>87131.0</td>
      <td>NaN</td>
      <td>332.0</td>
      <td>-1096.0</td>
      <td>1759.0</td>
    </tr>
    <tr>
      <th>2020-12-25</th>
      <td>NaN</td>
      <td>84393.0</td>
      <td>81346.0</td>
      <td>87439.0</td>
      <td>NaN</td>
      <td>308.0</td>
      <td>-1120.0</td>
      <td>1736.0</td>
    </tr>
    <tr>
      <th>2020-12-26</th>
      <td>NaN</td>
      <td>84679.0</td>
      <td>81633.0</td>
      <td>87726.0</td>
      <td>NaN</td>
      <td>286.0</td>
      <td>-1141.0</td>
      <td>1714.0</td>
    </tr>
    <tr>
      <th>2020-12-27</th>
      <td>NaN</td>
      <td>84945.0</td>
      <td>81899.0</td>
      <td>87992.0</td>
      <td>NaN</td>
      <td>266.0</td>
      <td>-1162.0</td>
      <td>1694.0</td>
    </tr>
    <tr>
      <th>2020-12-28</th>
      <td>NaN</td>
      <td>85193.0</td>
      <td>82146.0</td>
      <td>88239.0</td>
      <td>NaN</td>
      <td>247.0</td>
      <td>-1181.0</td>
      <td>1675.0</td>
    </tr>
    <tr>
      <th>2020-12-29</th>
      <td>NaN</td>
      <td>85422.0</td>
      <td>82376.0</td>
      <td>88468.0</td>
      <td>NaN</td>
      <td>229.0</td>
      <td>-1198.0</td>
      <td>1657.0</td>
    </tr>
    <tr>
      <th>2020-12-30</th>
      <td>NaN</td>
      <td>85634.0</td>
      <td>82587.0</td>
      <td>88680.0</td>
      <td>NaN</td>
      <td>212.0</td>
      <td>-1216.0</td>
      <td>1640.0</td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>NaN</td>
      <td>85829.0</td>
      <td>82783.0</td>
      <td>88876.0</td>
      <td>NaN</td>
      <td>196.0</td>
      <td>-1232.0</td>
      <td>1623.0</td>
    </tr>
  </tbody>
</table>
</div>

from the above results we can see that our model predicts the total number of
confirmed cases with an accuracy of 97.99%,

On the 11th of December the model predicted 77603 cases which is off compared to
the actual number of cases on that day (80k). Based on this, we can conclude
that there is always going to be some difference between the prediction and
actual data, so we try to extract a range in which the number of cases will
fall. Below is the graph for predicted daily cases range.

```python
plot_results(my_lstm_df, "Malaysia", "LSTM")
```

![png](/assets/images/covid-19-forecasting-malaysia-data-analysis_files/covid-19-forecasting-malaysia-data-analysis_39_0.png)

### Question 3: how many cases will Malaysia have by the end of 2020?

based on the above graph and previous dataframe output, we can conclude that by
the end of 2020 Malaysia will have a total number of confirmed cases between 82k
and 88k. We should take the predicted daily cases with a grain of salt since the
model is linear, it cannot predict non-linear values.

### Question 4: how can we make the number of daily cases go down before the end of 2020?

To answer this question, we have to look back at how did we reduce the number of
daily cases and stabilize the number of total confirmed cases in the first wave;
the answer is mandated lockdowns.

# Effectiveness of mandated lockdown

Malaysia lockdown started from mid march and some might argue its still ongoing,
however the currently ongoing variations of the lockdown are not as strict as
the first lockdown and are less effective, here I try to show the effect the
first strict lockdown had on the total number of cases, to further support the
answer to question 4.

**restricted lockdown Time frame**

from 2020-03-16 until 2020-05-04

```python
# generate a dataframe with given range
def get_range_df(start: str, end: str, df):
    target_df = df.loc[pd.to_datetime(start, format='%Y-%m-%d'):pd.to_datetime(end, format='%Y-%m-%d')]
    return target_df
```

```python
my_lockdown = get_range_df('2020-03-16', '2020-05-04', my_con_series)
my_pre = get_range_df('2020-01-25', '2020-03-16', my_con_series)
my_after = get_range_df('2020-05-04', '2020-12-11', my_con_series)
```

```python


plt.figure(figsize=(18,9))
plt.plot(my_lockdown, color="C7", label="during lockdown", linewidth=5)
plt.plot(my_pre, color="C8", label="before lockdown")
plt.plot(my_after, color="C3", label="after lockdown")
plt.legend()
plt.show()
```

![png](/assets/images/covid-19-forecasting-malaysia-data-analysis_files/covid-19-forecasting-malaysia-data-analysis_45_0.png)

from the graph we can see that the number of total cases stabilized after the
lockdown, meaning the number of daily cases was so low its not even visible on a
graph, and this shows the effect the restricted lockdown had on the cases
further proving the answer to question 4 to be correct.
