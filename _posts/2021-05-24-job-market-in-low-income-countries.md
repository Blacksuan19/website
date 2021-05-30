---
title: Job Market In Low Income Countries
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/job-market-in-low-income-countries
tags:
  - data-science
  - machine-learning
  - project
---

## Questions

the focus is to obtain some specific data on skill migration among low income
countries, and to predict the skill migration trends in each country for 2020

- list of countries classified as low income by the world bank
- which skill group category had positive migration in 2019
- which industry and country had the most positive migration in 2019
- list of countries with positive skill migration in 2019
- skill migration in countries with more the 1k per 10k in 2019 vs 2015
- predict skill migration per 10k for 2020
- which country will have the most skill migration in skills that had positive
  migration in 2019 for 2020

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import Holt

# supress annoying warning
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

# set the size for all figures
plt.rcParams['figure.figsize'] = [14, 7]
```

```python
# load in the dataset
talent_mg_skill = pd.read_excel("../input/linkedin-digital-data/public_use-talent-migration.xlsx", sheet_name="Skill Migration")
```

```python
# get countries with low income
countries_low = talent_mg_skill[talent_mg_skill["wb_income"] == "Low income"]
```

```python
print("list of countries classified as low income:")
for country in countries_low["country_name"].unique():
    print(country)
```

    list of countries classified as low income:
    Afghanistan
    Benin
    Burkina Faso
    Congo, Dem. Rep.
    Ethiopia
    Haiti
    Madagascar
    Malawi
    Mali
    Mozambique
    Nepal
    Rwanda
    Senegal
    Syrian Arab Republic
    Tanzania
    Togo
    Uganda
    Yemen, Rep.
    Zimbabwe

```python
# get industries which had positive migration in 2019
pos_2019 = countries_low[countries_low["net_per_10K_2019"] > 0]
```

```python
print("list of skill group with positive migration in 2019:")
for group in pos_2019["skill_group_category"].unique():
    print(group)

```

    list of skill group with positive migration in 2019:
    Business Skills
    Specialized Industry Skills
    Tech Skills
    Disruptive Tech Skills
    Soft Skills

```python
pos_2019[pos_2019["net_per_10K_2019"] == pos_2019["net_per_10K_2019"].max()]
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
      <th>country_code</th>
      <th>country_name</th>
      <th>wb_income</th>
      <th>wb_region</th>
      <th>skill_group_id</th>
      <th>skill_group_category</th>
      <th>skill_group_name</th>
      <th>net_per_10K_2015</th>
      <th>net_per_10K_2016</th>
      <th>net_per_10K_2017</th>
      <th>net_per_10K_2018</th>
      <th>net_per_10K_2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>107</th>
      <td>af</td>
      <td>Afghanistan</td>
      <td>Low income</td>
      <td>South Asia</td>
      <td>827</td>
      <td>Specialized Industry Skills</td>
      <td>Automotive</td>
      <td>-726.39</td>
      <td>-99.97</td>
      <td>-479.23</td>
      <td>-126.45</td>
      <td>518.73</td>
    </tr>
  </tbody>
</table>
</div>

the automotive industry in afghanstan had the most growth in 2019 compared to
other skill groups, from this we can also infer that the Specialized Industry
Skills category had the most growth of all groups in 2019

```python
# group rows by country
country_mg_2019 = pos_2019.groupby("country_name").sum()
```

```python
# lets take a look at each country in numbers
country_mg_2019
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
      <th>skill_group_id</th>
      <th>net_per_10K_2015</th>
      <th>net_per_10K_2016</th>
      <th>net_per_10K_2017</th>
      <th>net_per_10K_2018</th>
      <th>net_per_10K_2019</th>
    </tr>
    <tr>
      <th>country_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>10549</td>
      <td>-2487.43</td>
      <td>-1295.07</td>
      <td>-1406.04</td>
      <td>284.92</td>
      <td>585.78</td>
    </tr>
    <tr>
      <th>Burkina Faso</th>
      <td>4349</td>
      <td>-303.68</td>
      <td>475.66</td>
      <td>-63.77</td>
      <td>-96.72</td>
      <td>140.30</td>
    </tr>
    <tr>
      <th>Congo, Dem. Rep.</th>
      <td>267110</td>
      <td>15921.11</td>
      <td>5191.17</td>
      <td>-608.96</td>
      <td>569.27</td>
      <td>4348.39</td>
    </tr>
    <tr>
      <th>Ethiopia</th>
      <td>43143</td>
      <td>653.31</td>
      <td>-207.12</td>
      <td>-1726.96</td>
      <td>-1796.52</td>
      <td>554.37</td>
    </tr>
    <tr>
      <th>Madagascar</th>
      <td>34054</td>
      <td>790.39</td>
      <td>-220.53</td>
      <td>-1227.28</td>
      <td>-703.36</td>
      <td>322.82</td>
    </tr>
    <tr>
      <th>Malawi</th>
      <td>112564</td>
      <td>3260.79</td>
      <td>4743.21</td>
      <td>-208.35</td>
      <td>-597.27</td>
      <td>618.25</td>
    </tr>
    <tr>
      <th>Mali</th>
      <td>142676</td>
      <td>9017.98</td>
      <td>5482.09</td>
      <td>2333.82</td>
      <td>532.32</td>
      <td>2081.85</td>
    </tr>
    <tr>
      <th>Mozambique</th>
      <td>237709</td>
      <td>5390.04</td>
      <td>-5692.43</td>
      <td>-6644.60</td>
      <td>-2357.83</td>
      <td>4048.02</td>
    </tr>
    <tr>
      <th>Nepal</th>
      <td>36235</td>
      <td>417.14</td>
      <td>288.91</td>
      <td>-178.06</td>
      <td>-101.40</td>
      <td>597.68</td>
    </tr>
    <tr>
      <th>Rwanda</th>
      <td>142192</td>
      <td>1884.73</td>
      <td>1484.01</td>
      <td>919.05</td>
      <td>-2263.96</td>
      <td>3187.22</td>
    </tr>
    <tr>
      <th>Senegal</th>
      <td>203091</td>
      <td>3569.81</td>
      <td>3784.49</td>
      <td>-181.22</td>
      <td>224.01</td>
      <td>5335.45</td>
    </tr>
    <tr>
      <th>Syrian Arab Republic</th>
      <td>133</td>
      <td>-513.09</td>
      <td>-759.56</td>
      <td>-457.79</td>
      <td>-243.59</td>
      <td>9.12</td>
    </tr>
    <tr>
      <th>Tanzania</th>
      <td>83596</td>
      <td>453.32</td>
      <td>556.95</td>
      <td>-365.45</td>
      <td>-671.36</td>
      <td>284.23</td>
    </tr>
    <tr>
      <th>Togo</th>
      <td>559</td>
      <td>-113.94</td>
      <td>190.67</td>
      <td>-234.12</td>
      <td>-283.07</td>
      <td>9.87</td>
    </tr>
    <tr>
      <th>Uganda</th>
      <td>9268</td>
      <td>107.40</td>
      <td>125.65</td>
      <td>-252.66</td>
      <td>6.44</td>
      <td>261.38</td>
    </tr>
    <tr>
      <th>Yemen, Rep.</th>
      <td>9590</td>
      <td>-1586.55</td>
      <td>-1486.05</td>
      <td>-522.68</td>
      <td>164.03</td>
      <td>324.38</td>
    </tr>
  </tbody>
</table>
</div>

```python
country_mg_2019["net_per_10K_2019"]
```

    country_name
    Afghanistan              585.78
    Burkina Faso             140.30
    Congo, Dem. Rep.        4348.39
    Ethiopia                 554.37
    Madagascar               322.82
    Malawi                   618.25
    Mali                    2081.85
    Mozambique              4048.02
    Nepal                    597.68
    Rwanda                  3187.22
    Senegal                 5335.45
    Syrian Arab Republic       9.12
    Tanzania                 284.23
    Togo                       9.87
    Uganda                   261.38
    Yemen, Rep.              324.38
    Name: net_per_10K_2019, dtype: float64

lets plot countries which have more than 1000 migration on every 10k

```python
country_mg_2019[country_mg_2019["net_per_10K_2019"] > 1000].plot(y=["net_per_10K_2019"], style=".")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fde87f1d490>

![png](/assets/images/job-market-in-low-income-countries_files/job-market-in-low-income-countries_14_1.png)

we can see that sengal had the most skill migration in 2019 compared to other
countries now lets compare these numbers to 2015 for example

```python
country_mg_2019[country_mg_2019["net_per_10K_2019"] > 1000].plot(y=["net_per_10K_2019", "net_per_10K_2015"], style=".")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fde8f0aef10>

![png](/assets/images/job-market-in-low-income-countries_files/job-market-in-low-income-countries_16_1.png)

skill migration in countries with more than 1000 per 10k skill migration has
drastically changes compared to 2015, for example in the Congo and Mali people
are far less likely to migrate industries in 2019 compared to 2015 this could
indicate a stability in the job market and that people are now settling to a
specific field, in contrast more people are migrating to other industries in
2019 compared to 2015 in Senegal indicating a shift in the job market

```python
'''
Holt's (Method) Exponential Smoothing for predicting next value based on previous years values,
also known as forecasting
'''

def hes_predict(train):
    model = Holt(train)
    model_fit = model.fit()
    fcast = model_fit.forecast()
    return fcast[0]
```

```python
countries_2020 = pd.DataFrame(columns = ["net_per_10k_2020"], index = country_mg_2019.index)

for country in country_mg_2019.index:

    # take previous numbers for country as model input
    train = country_mg_2019.drop("skill_group_id", axis=1)[country_mg_2019.index == f"{country}"].to_numpy()

    # get model prediciton and round to 2 decimal places
    result = round(hes_predict(train[0]), 2)

    # save model prediction to dataframe
    countries_2020["net_per_10k_2020"][f"{country}"] = result

    # print prediction results
    print(f"{country} skill migration per 10k for 2020 = {result}\n")

```

    Afghanistan skill migration per 10k for 2020 = 566.64
    Burkina Faso skill migration per 10k for 2020 = 69.35
    Congo, Dem. Rep. skill migration per 10k for 2020 = 5297.78
    Ethiopia skill migration per 10k for 2020 = 2904.68
    Madagascar skill migration per 10k for 2020 = 1349.0
    Malawi skill migration per 10k for 2020 = 1700.94
    Mali skill migration per 10k for 2020 = -893.72
    Mozambique skill migration per 10k for 2020 = 9463.79
    Nepal skill migration per 10k for 2020 = 359.15
    Rwanda skill migration per 10k for 2020 = 1042.21
    Senegal skill migration per 10k for 2020 = 2587.35
    Syrian Arab Republic skill migration per 10k for 2020 = 281.75
    Tanzania skill migration per 10k for 2020 = 51.54
    Togo skill migration per 10k for 2020 = 0.1
    Uganda skill migration per 10k for 2020 = 101.82
    Yemen, Rep. skill migration per 10k for 2020 = 389.26

```python
# again plot countries where its more than 1k
countries_2020[countries_2020["net_per_10k_2020"] > 1000].plot(style=".")
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7fde87fa2210>

![png](/assets/images/job-market-in-low-income-countries_files/job-market-in-low-income-countries_20_1.png)

from the model predicitions we can see that the job market will have a major
shift in 2020 in Mozambique, with almost all 10k shifting between skills,
previous year's leading to a very unstable job market high, last years highest
migration was in Senegal with over 5k migrants per 10k, for 2020 the market
seems to be stabilizing and people are settling into jobs leading to less than
3k migrants.

in conclusion. the market has changed significantly for the past years in low
income countries, some of the countries had positive skill migration meaning an
unstable market where workers do not settle for a specific field, while other
countries had negative migration compared to previous years leading to a more
stable market which can be interpreted as a good measure of market stability and
quality of work for those workers
