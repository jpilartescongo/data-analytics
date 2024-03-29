# import auxiliary packages/modules
import seaborn as sns
import os, pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -------------------------------------------------------------
# set up function to create generic plot for input datasets
def generic_plot(data, x, y, color, label, title, ylabel):
  plt.plot(data[x], data[y], color=color, label=label)
  plt.title(title, fontsize=11)
  plt.ylabel(ylabel)
  plt.legend()
  plt.ylim(-0.5, 1)
  plt.grid(True)
  plt.tight_layout()

# -------------------------------------------------------------
# function to check if dataset is normally distributed 
# uses a given dataset in pandas pf format
# uses string that represents attribute of interest
def check_distribution(dataset, attr):
  stat, p = stats.kstest(dataset[attr], 'norm', (dataset[attr].mean(), dataset[attr].std()))

  # print the results
  print('K-S test statistic:', stat)
  print('p-value:', p)

  # interpret the results
  if p > 0.05:
    print('The data is normally distributed.')
  else:
    print('The data is not normally distributed.')

# -------------------------------------------------------------
# get the count of nans for various variables
def get_count_of_nans(df, vars):
  nan_counts = df[vars].isna().sum()
  nan_counts_df = pd.DataFrame(nan_counts, columns=['Count'])
  return nan_counts_df

# -------------------------------------------------------------
# function that finds remaining missing values from wind speed
# and wind gust columns and replace them with daily mean
def fill_nans(dataframe):
  # iterate through each row in the data frame
  for index, row in dataframe.iterrows():
    date = row['date']
    wind_speed_nan = pd.isna(row['wind_speed'])
    wind_gust_nan = pd.isna(row['wind_gust'])
    water_temp_nan = pd.isna(row['water_temp'])

    # check for are missing wind_speed, wind_gust, or water_temp values
    if wind_speed_nan or wind_gust_nan or water_temp_nan:
      # Filter the corresponding records for the specific date
      same_day_records = dataframe[dataframe['date'] == date]

      # calculate the average for 'wind_speed', 'wind_gust', and 'water_temp' on that day
      avg_wind_speed = same_day_records['wind_speed'].mean()
      avg_wind_gust = same_day_records['wind_gust'].mean()
      avg_water_temp = same_day_records['water_temp'].mean()

      # replace NaN values with the calculated averages
      if wind_speed_nan:
        dataframe.at[index, 'wind_speed'] = avg_wind_speed
      if wind_gust_nan:
        dataframe.at[index, 'wind_gust'] = avg_wind_gust
      if water_temp_nan:
        dataframe.at[index, 'water_temp'] = avg_water_temp

  # reset index after modification
  dataframe.reset_index(drop=True, inplace=True)
  return dataframe
  
# -------------------------------------------------------------
# function to create line graph using wind speed, wind gust, 
# and water levels for tide stations
def create_linegraph(dataframe, title):
  # plot showing the relationship between
  # water level wind speed, and wind gusts
  fig, ax1 = plt.subplots(figsize=(13, 5))

  # primary y axis (wind speed and wind gust)
  dataframe['wind_speed'].plot(kind='line', ax=ax1, color='coral', label='Wind Speed')
  dataframe['wind_gust'].plot(kind='line', ax=ax1, color='darkorchid', label='Wind Gust')
  ax1.set_xlabel('Date')
  ax1.set_ylabel('Wind Gust & Speed (m/s)', color='black')
  ax1.tick_params(axis='y', labelcolor='black')

  # secondary y axis (wind speed and wind gust)
  ax2 = ax1.twinx()
  dataframe['water_level'].plot(kind='line', linestyle='--', ax=ax2, color='black', linewidth=1, label='Water Level')
  ax2.set_ylabel('Water Level (m) - MLLW', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  # customize the resulting graph
  lines, labels = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(lines + lines2, labels + labels2, loc='upper left')
  plt.title(title)
  plt.show()
  
# -------------------------------------------------------------
# function to create hexbin
def create_hexbin(x, y, bar_label, x_label, y_label):
  fig, ax = plt.subplots()
  hexbin = ax.hexbin(x, y, gridsize=20, cmap='Blues')
  cbar = plt.colorbar(hexbin, label=bar_label)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

# -------------------------------------------------------------
# function to create 8 correlation plots
def create_corr_plots(dataframe, plot_title):
  xlabels = ['Wind Speed (m/s)', 'Wind Gust (m/s)', 'Wind Direction (º)', 'Water Temperature (ºC)', 'Wind Speed (m/s)', 'Wind Speed (m/s)']
  ylabels = ['Water Level (m) - MLLW', 'Water Level (m) - MLLW', 'Water Level (m) - MLLW', 'Water Level (m) - MLLW', 'Water Temperature (ºC)', 'Wind Gust (m/s)']

  fig, axes = plt.subplots(2, 3, figsize=(15, 6))

  combinations = [('wind_speed', 'wind_gust'), ('water_level', 'wind_speed'), ('water_level', 'wind_gust'), 
          ('water_level', 'wind_dir'), ('water_level', 'water_temp'), ('wind_speed', 'water_temp')]

  for pair, ax, xlabel, ylabel in zip(combinations, axes.flatten(), xlabels, ylabels):
    sns.regplot(data=dataframe, x=pair[0], y=pair[1], ax=ax)
    correlation_coefficient, _ = pearsonr(dataframe[pair[0]], dataframe[pair[1]])
    ax.text(0.5, 0.95, f"R = {correlation_coefficient:.2f}", ha='center', va='top', transform=ax.transAxes, fontsize=10, fontweight='bold', color='red')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

  plt.suptitle(plot_title)
  plt.tight_layout()
  























