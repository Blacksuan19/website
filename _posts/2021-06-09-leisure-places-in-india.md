---
title: Leisure Places In India
description: Data Science Project
layout: post
project: true
permalink: "/projects/:title/"
image: /assets/images/ds.jpg
source: https://www.kaggle.com/abubakaryagob/leisure-places-in-india
tags:
  - data-science
  - machine-learning
  - project
---

### Goals

- show places of leisure in a map
- make observations about the places and their distribution
- explore the most widely available leisure type in india

```python
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import ast
import plotly.express as px
import geopandas as gpd
```

```python
le = pd.read_csv("../input/buildings-amenities-all-over-india/leisure.csv")
```

```python
le.head()
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
      <th>Unnamed: 0</th>
      <th>name</th>
      <th>leisure</th>
      <th>longitude-lattitude</th>
      <th>All_tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>249132377</td>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>(77.10471029999984, 28.45473270000001)</td>
      <td>{'name': 'DLF Golf Links Golf Course', 'barrie...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>250737365</td>
      <td>NaN</td>
      <td>park</td>
      <td>(80.23786640000002, 13.04278489999996)</td>
      <td>{'leisure': 'park'}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>250979543</td>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>(75.8870475, 31.52995199999996)</td>
      <td>{'name': 'Yoga Centre', 'leisure': 'sports_cen...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>280167017</td>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>(76.9132247999999, 11.32635400000001)</td>
      <td>{'name': 'Black Thunder', 'leisure': 'water_pa...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>280701513</td>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>(76.67157809999996, 11.417312599999995)</td>
      <td>{'name': 'Ootacamund Gymkhana Golf Course', 'l...</td>
    </tr>
  </tbody>
</table>
</div>

```python
# remove all tags column and rename columns
le = le.drop("All_tags", axis=1)
le.columns = ["id", "name", "leisure", "lo-la"]
```

```python
le.head()
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
      <th>name</th>
      <th>leisure</th>
      <th>lo-la</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>249132377</td>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>(77.10471029999984, 28.45473270000001)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>250737365</td>
      <td>NaN</td>
      <td>park</td>
      <td>(80.23786640000002, 13.04278489999996)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>250979543</td>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>(75.8870475, 31.52995199999996)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>280167017</td>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>(76.9132247999999, 11.32635400000001)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>280701513</td>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>(76.67157809999996, 11.417312599999995)</td>
    </tr>
  </tbody>
</table>
</div>

```python
# set the ID as index
le.index = le["id"]
le = le.drop("id", axis = 1)
```

```python
le.head()
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
      <th>name</th>
      <th>leisure</th>
      <th>lo-la</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249132377</th>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>(77.10471029999984, 28.45473270000001)</td>
    </tr>
    <tr>
      <th>250737365</th>
      <td>NaN</td>
      <td>park</td>
      <td>(80.23786640000002, 13.04278489999996)</td>
    </tr>
    <tr>
      <th>250979543</th>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>(75.8870475, 31.52995199999996)</td>
    </tr>
    <tr>
      <th>280167017</th>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>(76.9132247999999, 11.32635400000001)</td>
    </tr>
    <tr>
      <th>280701513</th>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>(76.67157809999996, 11.417312599999995)</td>
    </tr>
  </tbody>
</table>
</div>

```python
# check NA values
le.isna().sum()
```

    name       27143
    leisure        0
    lo-la      37876
    dtype: int64

the latitude and longitude are the most important columns so we will drop all
rows that do not have them

```python
le = le[le['lo-la'].notna()]
le
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
      <th>name</th>
      <th>leisure</th>
      <th>lo-la</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249132377</th>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>(77.10471029999984, 28.45473270000001)</td>
    </tr>
    <tr>
      <th>250737365</th>
      <td>NaN</td>
      <td>park</td>
      <td>(80.23786640000002, 13.04278489999996)</td>
    </tr>
    <tr>
      <th>250979543</th>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>(75.8870475, 31.52995199999996)</td>
    </tr>
    <tr>
      <th>280167017</th>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>(76.9132247999999, 11.32635400000001)</td>
    </tr>
    <tr>
      <th>280701513</th>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>(76.67157809999996, 11.417312599999995)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8277782288</th>
      <td>NaN</td>
      <td>playground</td>
      <td>(76.29733219999959, 10.029497999999887)</td>
    </tr>
    <tr>
      <th>8280851413</th>
      <td>Gothuruth muzhiris park</td>
      <td>park</td>
      <td>(76.21773650000003, 10.190251200000016)</td>
    </tr>
    <tr>
      <th>8280851414</th>
      <td>Gothuruth Muzhiris park</td>
      <td>park</td>
      <td>(76.21771200000003, 10.190284000000016)</td>
    </tr>
    <tr>
      <th>8281209559</th>
      <td>Exalt Fitness Club Gym</td>
      <td>fitness_centre</td>
      <td>(72.56438300000039, 23.089663400000084)</td>
    </tr>
    <tr>
      <th>8281506191</th>
      <td>NaN</td>
      <td>playground</td>
      <td>(75.54409639999994, 11.927387099999967)</td>
    </tr>
  </tbody>
</table>
<p>5813 rows × 3 columns</p>
</div>

fill the missing names with the word "missing"

```python
le['name'].fillna("missing", inplace=True)
```

```python
le
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
      <th>name</th>
      <th>leisure</th>
      <th>lo-la</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249132377</th>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>(77.10471029999984, 28.45473270000001)</td>
    </tr>
    <tr>
      <th>250737365</th>
      <td>missing</td>
      <td>park</td>
      <td>(80.23786640000002, 13.04278489999996)</td>
    </tr>
    <tr>
      <th>250979543</th>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>(75.8870475, 31.52995199999996)</td>
    </tr>
    <tr>
      <th>280167017</th>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>(76.9132247999999, 11.32635400000001)</td>
    </tr>
    <tr>
      <th>280701513</th>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>(76.67157809999996, 11.417312599999995)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8277782288</th>
      <td>missing</td>
      <td>playground</td>
      <td>(76.29733219999959, 10.029497999999887)</td>
    </tr>
    <tr>
      <th>8280851413</th>
      <td>Gothuruth muzhiris park</td>
      <td>park</td>
      <td>(76.21773650000003, 10.190251200000016)</td>
    </tr>
    <tr>
      <th>8280851414</th>
      <td>Gothuruth Muzhiris park</td>
      <td>park</td>
      <td>(76.21771200000003, 10.190284000000016)</td>
    </tr>
    <tr>
      <th>8281209559</th>
      <td>Exalt Fitness Club Gym</td>
      <td>fitness_centre</td>
      <td>(72.56438300000039, 23.089663400000084)</td>
    </tr>
    <tr>
      <th>8281506191</th>
      <td>missing</td>
      <td>playground</td>
      <td>(75.54409639999994, 11.927387099999967)</td>
    </tr>
  </tbody>
</table>
<p>5813 rows × 3 columns</p>
</div>

most available types of leisure places according to type

```python
le["leisure"].value_counts()
```

    park                        1798
    playground                   773
    fitness_centre               580
    resort                       544
    pitch                        497
    sports_centre                495
    fishing                      238
    garden                       196
    stadium                      159
    swimming_pool                155
    dance                         58
    fitness_station               56
    nature_reserve                52
    water_park                    24
    marina                        20
    slipway                       16
    beach_resort                  14
    common                        13
    amusement_arcade              12
    track                         10
    yes                           10
    outdoor_seating                9
    golf_course                    8
    recreation_ground              6
    club                           6
    bandstand                      5
    bowling_alley                  5
    hackerspace                    5
    bird_hide                      4
    adult_gaming_centre            4
    sauna                          4
    picnic_table                   3
    swimming_area                  3
    firepit                        3
    horse_riding                   3
    cultural_centre                2
    gym                            2
    hot_spring                     2
    indoor_play                    2
    wildlife_hide                  2
    spa                            2
    Park in residential area       1
    aquarium                       1
    leisure                        1
    ground                         1
    Meeting_point                  1
    sports_hall                    1
    summer_camp                    1
    social_club                    1
    yoga                           1
    schoolyard                     1
    NITTE FOOTBALL STADIUM         1
    quary                          1
    yoga_centre                    1
    Name: leisure, dtype: int64

lets draw a graph for an easier understanding

```python
plt.rcParams['font.size'] = 10.0
plt.rcParams['figure.figsize'] = 20, 10
ax = sns.countplot(le['leisure'], palette="Blues_r", order=le.leisure.value_counts()[:20].index)

ax.set_title("Most Avaiable Leisure Places in India")
# rotate the names so they fit
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
```

![png](/assets/images/leisure-places-in-india_files/leisure-places-in-india_17_0.png)

parks are the most common type of leisure building in india

```python
# split coordinates
cords = list(le["lo-la"])
long = []
lat = []
for cord in cords:
    set_r = ast.literal_eval(cord)
    long.append(set_r[0])
    lat.append(set_r[1])

le["long"] = long
le["lat"] = lat
```

```python
le.head()
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
      <th>name</th>
      <th>leisure</th>
      <th>lo-la</th>
      <th>long</th>
      <th>lat</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249132377</th>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>(77.10471029999984, 28.45473270000001)</td>
      <td>77.104710</td>
      <td>28.454733</td>
    </tr>
    <tr>
      <th>250737365</th>
      <td>missing</td>
      <td>park</td>
      <td>(80.23786640000002, 13.04278489999996)</td>
      <td>80.237866</td>
      <td>13.042785</td>
    </tr>
    <tr>
      <th>250979543</th>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>(75.8870475, 31.52995199999996)</td>
      <td>75.887047</td>
      <td>31.529952</td>
    </tr>
    <tr>
      <th>280167017</th>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>(76.9132247999999, 11.32635400000001)</td>
      <td>76.913225</td>
      <td>11.326354</td>
    </tr>
    <tr>
      <th>280701513</th>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>(76.67157809999996, 11.417312599999995)</td>
      <td>76.671578</td>
      <td>11.417313</td>
    </tr>
  </tbody>
</table>
</div>

```python
# drop the old coordinates column
le = le.drop("lo-la", axis=1)
```

```python
le.head()
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
      <th>name</th>
      <th>leisure</th>
      <th>long</th>
      <th>lat</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249132377</th>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>77.104710</td>
      <td>28.454733</td>
    </tr>
    <tr>
      <th>250737365</th>
      <td>missing</td>
      <td>park</td>
      <td>80.237866</td>
      <td>13.042785</td>
    </tr>
    <tr>
      <th>250979543</th>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>75.887047</td>
      <td>31.529952</td>
    </tr>
    <tr>
      <th>280167017</th>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>76.913225</td>
      <td>11.326354</td>
    </tr>
    <tr>
      <th>280701513</th>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>76.671578</td>
      <td>11.417313</td>
    </tr>
  </tbody>
</table>
</div>

```python
# basic scatter plot of places
plt.scatter(x=le["long"], y=le["lat"])
plt.show()
```

![png](/assets/images/leisure-places-in-india_files/leisure-places-in-india_23_0.png)

above is the initial shape of the locations in the map based on their longitude
and latitude, we can already see that the shape looks like india meaning there
are many leisure places around the country

```python
# create and view geopandas dataframe
gdf = gpd.GeoDataFrame(
    le, geometry=gpd.points_from_xy(le.long, le.lat))
gdf
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
      <th>name</th>
      <th>leisure</th>
      <th>long</th>
      <th>lat</th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>249132377</th>
      <td>DLF Golf Links Golf Course</td>
      <td>golf_course</td>
      <td>77.104710</td>
      <td>28.454733</td>
      <td>POINT (77.10471 28.45473)</td>
    </tr>
    <tr>
      <th>250737365</th>
      <td>missing</td>
      <td>park</td>
      <td>80.237866</td>
      <td>13.042785</td>
      <td>POINT (80.23787 13.04278)</td>
    </tr>
    <tr>
      <th>250979543</th>
      <td>Yoga Centre</td>
      <td>sports_centre</td>
      <td>75.887047</td>
      <td>31.529952</td>
      <td>POINT (75.88705 31.52995)</td>
    </tr>
    <tr>
      <th>280167017</th>
      <td>Black Thunder</td>
      <td>water_park</td>
      <td>76.913225</td>
      <td>11.326354</td>
      <td>POINT (76.91322 11.32635)</td>
    </tr>
    <tr>
      <th>280701513</th>
      <td>Ootacamund Gymkhana Golf Course</td>
      <td>golf_course</td>
      <td>76.671578</td>
      <td>11.417313</td>
      <td>POINT (76.67158 11.41731)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8277782288</th>
      <td>missing</td>
      <td>playground</td>
      <td>76.297332</td>
      <td>10.029498</td>
      <td>POINT (76.29733 10.02950)</td>
    </tr>
    <tr>
      <th>8280851413</th>
      <td>Gothuruth muzhiris park</td>
      <td>park</td>
      <td>76.217737</td>
      <td>10.190251</td>
      <td>POINT (76.21774 10.19025)</td>
    </tr>
    <tr>
      <th>8280851414</th>
      <td>Gothuruth Muzhiris park</td>
      <td>park</td>
      <td>76.217712</td>
      <td>10.190284</td>
      <td>POINT (76.21771 10.19028)</td>
    </tr>
    <tr>
      <th>8281209559</th>
      <td>Exalt Fitness Club Gym</td>
      <td>fitness_centre</td>
      <td>72.564383</td>
      <td>23.089663</td>
      <td>POINT (72.56438 23.08966)</td>
    </tr>
    <tr>
      <th>8281506191</th>
      <td>missing</td>
      <td>playground</td>
      <td>75.544096</td>
      <td>11.927387</td>
      <td>POINT (75.54410 11.92739)</td>
    </tr>
  </tbody>
</table>
<p>5813 rows × 5 columns</p>
</div>

```python
# set mapbox acces token (required for drawing an interactive map)
px.set_mapbox_access_token("pk.eyJ1IjoiYmxhY2tzdWFuMTkiLCJhIjoiY2twcDdtaGc4MDZ6djJvczR0Ym9sa3pqNCJ9.gNL1mxeSmDi6hfgwxz2qRA")
```

```python
# generate and show points in map (its intractive!)
fig = px.scatter_geo(gdf,
                    lat=gdf.geometry.y,
                    lon=gdf.geometry.x,
                    hover_data=["name", "leisure"],
                    locationmode="country names"
                    )
fig.update_geos(fitbounds="locations") # zoom in to only india
fig.show()
# check the project source code for the interactive version of the map
```

![plotly](/assets/images/leisure-places-in-india_files/plotly.png)

### observations form the map

- most of the leisure places are located the the cost
- there are some obvious outlier locations that are probably fake
- the center of india has the least amount of leisure places
- most of the resorts are located on the western cost
- parks are the only leisure activity available all across the country
