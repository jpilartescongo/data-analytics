# import auxiliary packages/modules
import os, pandas as pd
import matplotlib.pyplot as plt

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
def create_linegraphs(dataframe, title):
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
  ax2.set_ylabel('Water Level (m)', color='black')
  ax2.tick_params(axis='y', labelcolor='black')

  # customize the resulting graph
  lines, labels = ax1.get_legend_handles_labels()
  lines2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(lines + lines2, labels + labels2, loc='upper left')
  plt.title(title)
  plt.show()
  



























