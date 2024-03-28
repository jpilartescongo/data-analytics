# import auxiliary packages/modules
import os, pandas as pd

# -----------------------------------------------------------
# get the count of nans for various variables
def get_count_of_nans(df, vars):
  nan_counts = df[vars].isna().sum()
  nan_counts_df = pd.DataFrame(nan_counts, columns=['Count'])
  return nan_counts_df

# -----------------------------------------------------------
# call function that finds remaining missing values from wind 
# speed and wind gust columns and replaces them with the mean 
# for that particular day
def fill_nans(dataframe):
  # iterate through each row in the dataframe
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
