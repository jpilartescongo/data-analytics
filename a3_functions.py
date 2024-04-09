# import auxiliary packages/modules
import seaborn as sns
import os, pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import periodogram
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, median_absolute_error

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
  ax1.set_xlabel('Date Index (July 2020)')
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
# function to create periodogram for nueces bay and eagle point
def create_periodogram(df_array):
  plt.figure(figsize=(12, 4))
  labels = ['Nueces Bay', 'Eagle Point']
  
  for i, dataset in enumerate(df_array, 1):
    frequency = 0.5
    # set frequency and pxx
    # pxx is the power spectral density
    frequencies, pxx = periodogram(dataset['water_level'], frequency)
    
    plt.subplot(1, 2, i)
    plt.semilogy(frequencies, pxx)
    plt.xlabel('Frequency (cycles/hour)')
    plt.ylabel('Power Spectral Density')
    plt.title(f'Periodogram of Water Level Variation ({labels[i-1]})')
    plt.grid(True)
    plt.ylim(1e-30, 1e3)
    
  plt.suptitle('Water Level Variation Periodograms (2020)')
  plt.tight_layout()

# -------------------------------------------------------------
# function to create correlation plots for nueces bay and eagle
# point using a variety of tide station variables
def create_corr_plots(dataframe, plot_title, figsize):
  xlabels = ['Wind Speed (m/s)', 'Water Temperature (ºC)', 'Wind Speed (m/s)', 'Wind Gust (m/s)']
  ylabels = ['Wind Gust (m/s)', 'Water Level (m) - MLLW', 'Water Level (m) - MLLW', 'Water Level (m) - MLLW']

  fig, axes = plt.subplots(2, 2, figsize=figsize)

  combinations = [('wind_speed', 'wind_gust'), ('water_temp', 'water_level'), ('wind_speed', 'water_level'), ('wind_gust', 'water_level')]

  for pair, ax, xlabel, ylabel in zip(combinations, axes.flatten(), xlabels, ylabels):
    sns.regplot(data=dataframe, x=pair[0], y=pair[1], ax=ax)
    correlation_coefficient, _ = pearsonr(dataframe[pair[0]], dataframe[pair[1]])
    ax.text(0.5, 0.95, f"R = {correlation_coefficient:.2f}", ha='center', va='top', transform=ax.transAxes, fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

  plt.suptitle(plot_title)
  plt.tight_layout()

# -------------------------------------------------------------
# function that creates plots showing the distribution of water
# level data for different stations according to day of week
def create_wl_plots(nb_july_2020, ep_july_2020, labels):
  fig, axes = plt.subplots(2, 2, figsize=(14, 6))

  datasets = [(nb_july_2020.sort_values('day'), labels[0]), 
              (ep_july_2020.sort_values('day'), labels[1]), 
              (nb_july_2020, labels[0]), 
              (ep_july_2020, labels[1])]

  for i, ((data, label), ax) in enumerate(zip(datasets, axes.flatten())):
    if i < 2:
      sns.boxenplot(data=data, x='day_of_week', y='water_level', ax=ax)
    else:
      sns.boxplot(data=data, x='day_of_week', y='water_level', ax=ax)
    ax.set_title(label)
    ax.set_xlabel('')
    ax.set_ylabel('Water Level (m) - MLLW')
    ax.set_ylim(-0.25, 1.6)

  fig.suptitle('Weekly Distribution of Water Levels (July 2020)')
  plt.tight_layout()
  plt.show()

# -------------------------------------------------------------
# split time frames into date, time, month, and day of the week
def time_split(dataframe):
  dataframe['date'] = dataframe['date_time'].dt.date
  dataframe['time'] = dataframe['date_time'].dt.time
  month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
  dataframe['month'] = dataframe['date_time'].dt.month.map(month_names)
  dataframe['day_of_week'] = dataframe['date_time'].dt.strftime('%A')

# -------------------------------------------------------------
# general two general plots at bob hall pier
def generate_plots(dataframe1, dataframe2):
  fig, axes = plt.subplots(1, 2, figsize=(16, 5))
  
  for i, df, title in zip(range(2), [dataframe1, dataframe2], ['Training Dataset', 'Test Dataset']):
    axes[i].plot(df['date'], df['water_level'])
    axes[i].set_title(f'Water Levels (in MLLW) on {title} - Based on 6-minute Averages')
    axes[i].set_ylabel('Water level (ft) - MLLW')
    axes[i].set_xlabel('Date (YYYY-MM)')
    axes[i].grid(True)

  plt.tight_layout()
  plt.show()

# -------------------------------------------------------------
# figure with four subplots showing the distribution of training 
# and test data points; training is according to month and test
# is according to day of week
def distribution_plots(training_df, test_df):
  fig, axes = plt.subplots(2, 2, figsize=(15, 7))

  # set global parameters
  var1, var2 = 'month', 'water_level'
  color, edge_clr = 'darkorchid', 'black'
  labels = ['Training Dataset (Jan 1, 2017 - Dec 31, 2019)', 'Test Dataset (Jan 1, 2020 - Oct 21, 2020)']

  for i, data in enumerate([training_df, test_df]):
      # create box plots (according to month) as top two figures
      sns.boxplot(data=data, x=var1, y=var2, whis=(0, 100), ax=axes[0][i])
      axes[0][i].set_ylim(-2, 6)
      axes[0][i].set_title(labels[i], fontsize=11)
      axes[0][i].set_xlabel('')
      axes[0][i].set_ylabel('Water Level (m) - MLLW')
      axes[0][i].tick_params(axis='x', rotation=45)

      # create violin plots (according to day of week) as bottom two figures
      sns.violinplot(data=data, x='day_of_week', y='water_level', hue='day_of_week', split=True, ax=axes[1][i])
      axes[1][i].set_title(labels[i])
      axes[1][i].set_xlabel('')
      axes[1][i].set_ylabel('Water Level (m) - MLLW')

  plt.suptitle('Water Levels (in MLLW) from Training and Testing Datasets', fontsize=13)
  plt.tight_layout()
  plt.show()

# -------------------------------------------------------------
# select all records that fulfill assignment expectations of 
# predicting records where time is 00:00:00 or 12:00:00
def select_records(dataframe):
  dataframe['date_time'] = pd.to_datetime(dataframe['date_time'])
  dataframe['time'] = dataframe['date_time'].dt.strftime('%H:%M:%S')
  selected_records = dataframe[(dataframe['time'] == '00:00:00') | (dataframe['time'] == '12:00:00')].copy()
  selected_records.drop(columns=['time'], inplace=True)

  return selected_records
  
# -------------------------------------------------------------
# function to calculate the statistics of model performance
def statistical_results(model_name, y_test, y_pred):
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  medae = median_absolute_error(y_test, y_pred)
  residuals = y_test - y_pred
  residual_std = np.std(residuals)
  maxerr = np.max(np.abs(residuals))
  
  print(model_name)
  print('Mean squared error (MSE):', mse)
  print('R-squared (R2):', r2)
  print('Mean absolute error (MAE):', mae)
  print('Root mean squared error (RMSE):', rmse)
  print('Median absolute error (MedAE):', medae)
  print('Residual standard deviation:', residual_std)
  print('Maximum error (MaxErr):', maxerr)

# -------------------------------------------------------------
# function to plot lstm loss graph based on learning history
def lstm_loss(history, title):
  train_loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 6))
  plt.plot(range(1, len(train_loss) + 1), train_loss, label='Loss (training)')
  plt.plot(range(1, len(val_loss) + 1), val_loss, label='Loss (validation)')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title(title)
  plt.legend()
  plt.show()


# -------------------------------------------------------------
# function to plot lstm prediction graph (pred vs actual)
def plot_lstm_prediction(y_pred, y_test, title):
  y_pred = pd.Series(y_pred.flatten(), index=y_test.index)
  plt.figure(figsize=(15, 5))
  plt.plot(y_pred, label='Water level (predicted)', color='darkorchid')
  plt.plot(y_test, label='Water level (actual)', color='coral')
  plt.title(title)
  plt.xlabel('Date (YYYY-MM)')
  plt.ylabel('Water Level (ft) - MLLW')
  plt.legend()
  plt.show()

# -------------------------------------------------------------
# function to plot rf prediction graph (pred vs actual)
def plot_rf_prediction(y_pred_rf, y_test, title):
  y_pred_rf = pd.Series(y_pred_rf.flatten(), index=y_test.index)
  plt.figure(figsize=(15, 5))
  plt.plot(y_pred_rf, label='Water level (predicted)', color='darkorchid')
  plt.plot(y_test, label='Water level (actual)', color='coral')
  plt.xlabel('Date (YYYY-MM)')
  plt.ylabel('Water Level (ft) - MLLW')
  plt.title(title)
  plt.legend()
  plt.show()

# -------------------------------------------------------------
# function to calculate the statistics of rf model performance
def rf_statistical_results(model_name, y_test, y_pred):
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  medae = median_absolute_error(y_test, y_pred)
  residuals = y_test - y_pred
  residual_std = np.std(residuals)
  maxerr = np.max(np.abs(residuals))
  
  print(model_name)
  print('Mean squared error (MSE):', mse)
  print('R-squared (R2):', r2)
  print('Mean absolute error (MAE):', mae)
  print('Root mean squared error (RMSE):', rmse)
  print('Median absolute error (MedAE):', medae)
  print('Residual standard deviation:', residual_std)
  print('Maximum error (MaxErr):', maxerr)
