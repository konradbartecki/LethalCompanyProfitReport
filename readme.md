# Lethal Company v50 - Profitability Report


```python
import random
import numpy as np
import pandas as pd
```


```python
df = pd.read_csv('moons.csv')
df['Mean Loot'] = (df['Min Scrap'].astype(int) + df['Max Scrap'].astype(int)) / 2
```


```python
import math


# Number of attempts
def select_random_items(data, random_items_n):
    spawn_chances = data['Spawn Chance'].str.replace('%', '').astype(float) / 100
    items = data['Item'].values
    random_items_n = math.floor(random_items_n)
    selected_items = random.choices(items, weights=spawn_chances, k=random_items_n)
    return selected_items

# Function to calculate the total based on average price of selected items
def calculate_total(data, selected_items):
    avg_prices = data.set_index('Item')['Average Value (c)'].astype(str).str.replace(' ▮', '').astype(float)
    total = sum(avg_prices.loc[item] for item in selected_items)
    return total

# Function to run the simulation 100 times and calculate the average total
def run_simulation(data, random_items_n, num_runs=100):
    totals = []
    for _ in range(num_runs):
        selected_items = select_random_items(data, random_items_n)
        total = calculate_total(data, selected_items)
        totals.append(total)
    avg_total = sum(totals) / num_runs
    return avg_total
```


```python
results = []

for _, moon in df.iterrows():
    moon_name = moon['Name'] + '.csv'
    moon_mean_items = moon['Mean Loot']
    item_data = pd.read_csv(moon_name)
    avg_total = run_simulation(item_data, moon_mean_items)
    avg_total_max_12 = run_simulation(item_data, min(moon_mean_items, 12))
    avg_total_max_16 = run_simulation(item_data, min(moon_mean_items, 16))
    avg_total_max_20 = run_simulation(item_data, min(moon_mean_items, 20))
    result = pd.DataFrame({
        'Moon': [moon['Name']],
        'Average Total (cents)': [avg_total],
        'Average Total with 12 items looted (cents)': [avg_total_max_12],
        'Average Total with 16 items looted (cents)': [avg_total_max_16],
        'Average Total with 20 items looted (cents)': [avg_total_max_20]
    })
    results.append(result)

combined_results = pd.concat(results, ignore_index=True)
```


```python
moons = df.join(combined_results.set_index('Moon'), on='Name')
moons['Average Total minus Cost'] = moons['Average Total (cents)'] - moons['Cost'].astype(float)
moons
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
      <th>Name</th>
      <th>Difficulty</th>
      <th>Cost</th>
      <th>Default Layout</th>
      <th>Map Size Multiplier</th>
      <th>Min Scrap</th>
      <th>Max Scrap</th>
      <th>Max Indoor Power</th>
      <th>Max Outdoor Power</th>
      <th>Mean Loot</th>
      <th>Average Total (cents)</th>
      <th>Average Total with 12 items looted (cents)</th>
      <th>Average Total with 16 items looted (cents)</th>
      <th>Average Total with 20 items looted (cents)</th>
      <th>Average Total minus Cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41-Experimentation</td>
      <td>Easy</td>
      <td>0</td>
      <td>The Factory</td>
      <td>1.00</td>
      <td>8</td>
      <td>11</td>
      <td>4</td>
      <td>8</td>
      <td>9.5</td>
      <td>280.73</td>
      <td>280.99</td>
      <td>275.55</td>
      <td>276.43</td>
      <td>280.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>220-Assurance</td>
      <td>Easy</td>
      <td>0</td>
      <td>The Factory</td>
      <td>1.00</td>
      <td>13</td>
      <td>15</td>
      <td>6</td>
      <td>8</td>
      <td>14.0</td>
      <td>531.76</td>
      <td>457.05</td>
      <td>532.48</td>
      <td>531.37</td>
      <td>531.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56-Vow</td>
      <td>Easy</td>
      <td>0</td>
      <td>The Factory</td>
      <td>1.15</td>
      <td>12</td>
      <td>14</td>
      <td>7</td>
      <td>6</td>
      <td>13.0</td>
      <td>489.50</td>
      <td>455.48</td>
      <td>487.93</td>
      <td>494.40</td>
      <td>489.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21-Offense</td>
      <td>Intermediate</td>
      <td>0</td>
      <td>The Factory</td>
      <td>1.25</td>
      <td>14</td>
      <td>17</td>
      <td>12</td>
      <td>8</td>
      <td>15.5</td>
      <td>554.82</td>
      <td>446.37</td>
      <td>552.60</td>
      <td>548.99</td>
      <td>554.82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61-March</td>
      <td>Intermediate</td>
      <td>0</td>
      <td>The Factory</td>
      <td>2.00</td>
      <td>13</td>
      <td>16</td>
      <td>14</td>
      <td>12</td>
      <td>14.5</td>
      <td>544.35</td>
      <td>451.30</td>
      <td>539.16</td>
      <td>536.28</td>
      <td>544.35</td>
    </tr>
    <tr>
      <th>5</th>
      <td>20-Adamance</td>
      <td>Intermediate</td>
      <td>0</td>
      <td>The Factory</td>
      <td>1.18</td>
      <td>16</td>
      <td>18</td>
      <td>13</td>
      <td>13</td>
      <td>17.0</td>
      <td>674.21</td>
      <td>457.69</td>
      <td>610.15</td>
      <td>660.12</td>
      <td>674.21</td>
    </tr>
    <tr>
      <th>6</th>
      <td>85-Rend</td>
      <td>Hard</td>
      <td>550</td>
      <td>The Manor</td>
      <td>1.80</td>
      <td>18</td>
      <td>25</td>
      <td>10</td>
      <td>6</td>
      <td>21.5</td>
      <td>1229.86</td>
      <td>692.48</td>
      <td>926.15</td>
      <td>1151.87</td>
      <td>679.86</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7-Dine</td>
      <td>Hard</td>
      <td>600</td>
      <td>The Manor</td>
      <td>1.80</td>
      <td>22</td>
      <td>25</td>
      <td>16</td>
      <td>6</td>
      <td>23.5</td>
      <td>1273.28</td>
      <td>681.06</td>
      <td>903.92</td>
      <td>1122.88</td>
      <td>673.28</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8-Titan</td>
      <td>Hard</td>
      <td>700</td>
      <td>The Factory</td>
      <td>2.20</td>
      <td>28</td>
      <td>31</td>
      <td>18</td>
      <td>7</td>
      <td>29.5</td>
      <td>1426.27</td>
      <td>611.41</td>
      <td>805.23</td>
      <td>999.07</td>
      <td>726.27</td>
    </tr>
    <tr>
      <th>9</th>
      <td>68-Artifice</td>
      <td>Hard</td>
      <td>1500</td>
      <td>The Manor</td>
      <td>1.60</td>
      <td>31</td>
      <td>37</td>
      <td>13</td>
      <td>13</td>
      <td>34.0</td>
      <td>1991.77</td>
      <td>717.06</td>
      <td>935.80</td>
      <td>1176.72</td>
      <td>491.77</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(moons['Map Size Multiplier'], moons['Average Total (cents)'], c=moons['Max Outdoor Power'], s=moons['Max Indoor Power'] * 10, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, label='Max Outdoor Power')
plt.xlabel('Map Size Multiplier')
plt.ylabel('Average Total (cents)')


for i, name in enumerate(moons['Name']):
    ax.annotate(name, (moons['Map Size Multiplier'].iloc[i], moons['Average Total (cents)'].iloc[i]))

plt.show()
```


    
![png](output_6_0.png)
    



```python
# Create the line chart
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(moons.index, moons['Map Size Multiplier'] * 10, marker='o', label='Map Size Multiplier * 10')


ax.plot(moons.index, moons['Average Total (cents)'] / 100, marker='s', label='Average Total (cents) / 100')
ax.plot(moons.index, moons['Average Total with 12 items looted (cents)'] / 100, marker='s', label='Average Total (12 looted)')
ax.plot(moons.index, moons['Average Total with 16 items looted (cents)'] / 100, marker='s', label='Average Total (16 looted)')
ax.plot(moons.index, moons['Average Total with 20 items looted (cents)'] / 100, marker='s', label='Average Total (20 looted))')


ax.plot(moons.index, moons['Max Outdoor Power'], marker='v', label='Max Outdoor Power')

ax.plot(moons.index, moons['Max Indoor Power'], marker='^', label='Max Indoor Power')

ax.set_xticks(np.arange(len(moons)))
ax.set_xticklabels(moons['Name'], rotation=45, ha='right')
ax.set_xlabel('Name')
ax.set_ylabel('Value')
ax.set_title('Line Chart')
ax.legend()

plt.show()
```


    
![png](output_7_0.png)
    



```python
fig, ax1 = plt.subplots(figsize=(12, 6))


ax1.bar(moons.index, moons['Max Outdoor Power'], color='b', alpha=0.6, label='Max Outdoor Power')
ax1.bar(moons.index, moons['Max Indoor Power'], color='g', alpha=0.6, bottom=moons['Max Outdoor Power'], label='Max Indoor Power')
ax1.set_ylabel('Power Levels')
ax1.set_xticks(np.arange(len(moons)))
ax1.set_xticklabels(moons['Name'], rotation=45)
ax1.legend(loc='upper left')

ax2 = ax1.twinx() 

colors = ['r', 'm', 'k', 'y', 'salmon']  
labels = ['Map Size Multiplier * 10', 'Avg Total/100', 'Avg Total (12 looted)/100', 'Avg Total (16 looted)/100', 'Cost/100']
scaling = [10, 0.01, 0.01, 0.01, 0.01]
columns = ['Map Size Multiplier', 'Average Total (cents)', 'Average Total with 12 items looted (cents)', 'Average Total with 16 items looted (cents)', 'Cost']

for color, label, scale, column in zip(colors, labels, scaling, columns):
    ax2.plot(moons.index, moons[column] * scale, color=color, marker='o', label=label)

ax2.set_ylabel('Scaled Values')
ax2.legend(loc='upper right')

plt.title('Combined Bar and Line Graph')
plt.show()
```

    
![png](output_10_0.png)

![png](output_8_1.png)
    



```python
# Normalize data  (higher is better)
for column in ["Average Total (cents)", "Average Total with 12 items looted (cents)",
               "Average Total with 16 items looted (cents)", "Average Total with 20 items looted (cents)"]:
    moons[column] = (moons[column] - moons[column].min()) / (moons[column].max() - moons[column].min())

# Normalize data (lower is better)
for column in ["Map Size Multiplier", "Max Indoor Power", "Max Outdoor Power", "Cost"]:
    moons[column] = 1 - ((moons[column] - moons[column].min()) / (moons[column].max() - moons[column].min()))

moons['Score'] = (moons["Average Total (cents)"] +
               moons["Average Total with 12 items looted (cents)"] +
               moons["Average Total with 16 items looted (cents)"] +
               moons["Average Total with 20 items looted (cents)"] +
               moons["Map Size Multiplier"] +
               moons["Max Indoor Power"] +
               moons["Max Outdoor Power"] -
               moons["Cost"])

moons = moons.sort_values(by='Score', ascending=False)

print(moons[['Name', 'Score']])
```

                     Name     Score
    9         68-Artifice  4.857143
    6             85-Rend  4.727553
    7              7-Dine  4.265633
    8             8-Titan  3.355947
    1       220-Assurance  2.794199
    2              56-Vow  2.746647
    3          21-Offense  2.196325
    5         20-Adamance  1.775281
    0  41-Experimentation  1.714286
    4            61-March  0.827752



```python

```
