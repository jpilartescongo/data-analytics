# install required modules
# import auxiliary libraries
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import matplotlib.colors as mcolors
import seaborn as sns
import scipy.stats as stats

# ------------------------------------------
# function to plot general reference map of focus areas
def general_reference_map(coords):
  
  # set WGS84 as coordinate system
  coords_crs = 'EPSG:4326'
  projection = ccrs.LambertConformal()
  
  # display general reference (gr) map
  fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
  
  # plot coordinates on the map and add annotations for each location
  ax.scatter(x=[coord[1] for coord in coords], y=[coord[0] for coord in coords], transform=ccrs.PlateCarree())
  for i, coord in enumerate(coords):
    ax.text(coord[1], coord[0], coord[2], transform=ccrs.PlateCarree())
  
  # add gridlines and coastline boundaries
  ax.gridlines(draw_labels=True, color='none')
  ax.coastlines()
  
  # set map extent to area of interest
  states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces", "10m", facecolor="none", edgecolor="silver")
  ax.add_feature(states)
  ax.set_extent([-0.8e6, 0.7e6, -1.45e6, -0.7e6], crs=projection)
  
  # Set the title and show the plot
  ax.set_title('Coordinates of Locations of Interest')
  plt.show()


# --------------------------------------------------------------------------
# function to group features according to location
def indiv_station(dataset, city_name):
  feature = dataset[dataset['NAME'].str.startswith(city_name)]
  return feature

# --------------------------------------------------------------------------
# function to create hexbin
def create_hexbin(x, y, bar_label, x_label, y_label):
  fig, ax = plt.subplots()
  hexbin = ax.hexbin(x, y, gridsize=20, cmap='Blues')
  cbar = plt.colorbar(hexbin, label=bar_label)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

# --------------------------------------------------------------------------
# function to create temperature maps
def plot_subplot(ax, lon, lat, data, cmap, title):
    im = ax.pcolormesh(lon, lat, data, cmap=cmap, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_extent([-100, -93, 25, 33])
    ax.set_title(title)
    return im

def plot_pts(ax, station_data):
    ax.scatter(station_data['LONGITUDE'], station_data['LATITUDE'], color='black', marker='*', s=20, label='Station')
    ax.legend()

def create_cbar(im, axes):
    cbar = plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.09)
    cbar.set_label('Temperature (ºC)')

def generate_density_plot(data, ax, x_label, title):
  sns.distplot(data, ax=ax)
  ax.set_xlabel(x_label)
  ax.set_title(title)


# --------------------------------------------------------------------------
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

# --------------------------------------------------------------------------
# plot fire histograms for portugal data

def create_fire_hist2(dataset, var1, var2, title1, title2):
  fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  hist_x_lim = (-100, 7050)
  hist_y_lim = (0, 12)
  mean_label_x = 4000
  mean_label_y = 10

  mean_value1 = dataset[var1].mean()
  dataset[var1].plot(kind='hist', bins=20, ax=axs[0], color='orangered', edgecolor='black', linewidth=0.5)
  axs[0].set_title(title1)
  axs[0].spines[['top', 'right']].set_visible(True)
  axs[0].set_xlabel('Tree Cover Loss (ha)')
  axs[0].set_ylabel('Frequency')
  axs[0].axvline(mean_value1, color='black', linestyle='dashed', linewidth=2, label='Mean')
  axs[0].set_xlim(hist_x_lim)
  axs[0].set_ylim(hist_y_lim)
  # add mean value as text
  axs[0].text(mean_label_x, mean_label_y, f'Mean Loss: {mean_value1:.2f} ha', ha='center')

  mean_value2 = dataset[var2].mean()
  dataset[var2].plot(kind='hist', bins=20, ax=axs[1], color='darkorchid', edgecolor='black', linewidth=0.5)
  axs[1].set_title(title2)
  axs[1].spines[['top', 'right']].set_visible(True)
  axs[1].set_xlabel('Tree Cover Loss (ha)')
  axs[1].set_ylabel('Frequency')
  axs[1].axvline(mean_value2, color='black', linestyle='dashed', linewidth=2, label='Mean')
  axs[1].set_xlim(hist_x_lim)
  axs[1].set_ylim(hist_y_lim)
  # add mean value as text
  axs[1].text(mean_label_x, mean_label_y, f'Mean Loss: {mean_value2:.2f} ha', ha='center')

  plt.tight_layout()
  plt.show()

# --------------------------------------------------------------------------
# function to plot bar charts for two cities / sets of data
def create_bar_charts2(dataset1, dataset2, title1, title2, xlabel, ylabel):
  # Plot for Coimbra
  fig, axs = plt.subplots(1, 2, figsize=(12, 5))

  # Plot bars
  axs[0].bar(dataset1['YEAR'], dataset1['LOSS_HA'], color='orchid', edgecolor='black', linewidth=0.5, label='General Loss')
  axs[0].bar(dataset1['YEAR'], dataset1['LOSS_FIRES_HA'], color='coral', edgecolor='black', linewidth=0.5, label='Loss Due to Fire')
  axs[0].set_title(title1)
  axs[0].set_xlabel(xlabel)
  axs[0].set_ylabel(ylabel)
  axs[0].legend(loc='upper left')  # Add legend to the top left

  # Add line plot for trend
  axs[0].plot(dataset1['YEAR'], dataset1['LOSS_HA'].rolling(window=3).mean(), color='black', linestyle='--', label='Moving Average (3Yr)')
  axs[0].legend(loc='upper left')

  # Plot for Porto
  axs[1].bar(dataset2['YEAR'], dataset2['LOSS_HA'], color='orchid', edgecolor='black', linewidth=0.5, label='General Loss')
  axs[1].bar(dataset2['YEAR'], dataset2['LOSS_FIRES_HA'], color='coral', edgecolor='black', linewidth=0.5, label='Loss Due to Fire')
  axs[1].set_title(title2)
  axs[1].set_xlabel(xlabel)
  axs[1].set_ylabel(ylabel)
  axs[1].legend(loc='upper left')  # Add legend to the top left

  # Add line plot for trend
  axs[1].plot(dataset2['YEAR'], dataset2['LOSS_HA'].rolling(window=3).mean(), color='black', linestyle='--', label='Moving Average (3Yr)')
  axs[1].legend(loc='upper left')

  plt.tight_layout()
  plt.show()

# --------------------------------------------------------------------------
# 2x2 plot of coimbra/porto trend analyses using line graphs
def create_port_plots4(coimbra_data, porto_data, merged_data):
  # create figure with 2x2 figure settings and define label and titles for each subplot
  fig, axes = plt.subplots(2, 2, figsize=(12, 7))
  title1 = 'Average Annual Temperature'
  title2 = 'Total Annual Precipitation'
  title3 = 'Tree Cover Loss Due to Fire'
  title4 = 'Tree Cover Loss (General and Due To Fire)'
  xlabel, y_labels = 'Year', ['Temperature (ºC)', 'Precipitation (mm)', 'Tree Cover Loss (ha)']
  legend_loc = 'upper right'

  # set variable names and other attributes for better optimization
  y_attr = ['TAVG', 'PRCP', 'LOSS_HA', 'LOSS_FIRES_HA']
  x_attr = 'YEAR'

  city1, city2 = 'Coimbra', 'Porto'
  colors = ['darkorchid', 'orangered']

  # first plot
  sns.lineplot(x=x_attr, y=y_attr[0], data=coimbra_data, label=city1, color=colors[0], ax=axes[0, 0])
  sns.lineplot(x=x_attr, y=y_attr[0], data=porto_data, label=city2, color=colors[1], ax=axes[0, 0], linestyle='--')
  axes[0, 0].set_title(title1)
  axes[0, 0].set_xlabel(xlabel)
  axes[0, 0].set_ylabel(y_labels[0])
  axes[0, 0].legend(loc=legend_loc)

  # second plot
  sns.lineplot(x=x_attr, y=y_attr[1], data=coimbra_data, label=city1, color=colors[0], ax=axes[0, 1])
  sns.lineplot(x=x_attr, y=y_attr[1], data=porto_data, label=city2, color=colors[1], ax=axes[0, 1], linestyle='--')
  axes[0, 1].set_title(title2)
  axes[0, 1].set_xlabel(xlabel)
  axes[0, 1].set_ylabel(y_labels[1])
  axes[0, 1].legend(loc=legend_loc)

  # third plot
  sns.lineplot(x=x_attr, y=y_attr[3], data=coimbra_data, label=city1, color=colors[0], ax=axes[1, 0])
  sns.lineplot(x=x_attr, y=y_attr[3], data=porto_data, label=city2, color=colors[1], ax=axes[1, 0], linestyle='--')
  axes[1, 0].set_title(title3)
  axes[1, 0].set_xlabel(xlabel)
  axes[1, 0].set_ylabel(y_labels[2])
  axes[1, 0].legend(loc=legend_loc)

  # fourth plot
  sns.lineplot(x=x_attr, y=y_attr[2], data=merged_data, label='General Loss', color=colors[0], ax=axes[1, 1])
  sns.lineplot(x=x_attr, y=y_attr[3], data=merged_data, label='Loss Due to Fire', color=colors[1], ax=axes[1, 1], linestyle='--')
  axes[1, 1].set_title(title4)
  axes[1, 1].set_xlabel(xlabel)
  axes[1, 1].set_ylabel(y_labels[2])
  axes[1, 1].legend(loc=legend_loc)

  # adjust layout before display
  plt.tight_layout()
  plt.show()
  
# --------------------------------------------------------------------------
# 1x2 plot comparing temperature and precipitation in coimbra and porto
def create_weather_comparison(coimbra_merged, porto_merged):
  # extract data from coimbra_merged and porto
  yrs_coimbra = coimbra_merged['YEAR'].to_numpy()
  precip_coimbra = coimbra_merged['PRCP'].to_numpy()
  temp_coimbra = coimbra_merged['TAVG'].to_numpy()
  precip_error_coimbra = coimbra_merged['PRCP'].std()
  temp_error_coimbra = coimbra_merged['TAVG'].std()

  yrs_porto = porto_merged['YEAR'].to_numpy()
  precip_porto = porto_merged['PRCP'].to_numpy()
  temp_porto = porto_merged['TAVG'].to_numpy()
  precip_error_porto = porto_merged['PRCP'].std()
  temp_error_porto = porto_merged['TAVG'].std()

  # ----------------------------------------------------------------
  # create the precipitation bar chart for coimbra and porto
  plt.subplot(1, 2, 1)
  plt.bar(yrs_coimbra, temp_coimbra, color='darkorchid',
          yerr=temp_error_coimbra, align='center', ecolor='black',
          capsize=10, label='Coimbra', width= 0.25)

  plt.bar(yrs_porto +  0.25, temp_porto, color='coral',
          yerr=temp_error_porto, align='center', ecolor='black',
          capsize=10, label='Porto', width= 0.25)

  # set labels, title, y-axis limit, legend, and figure size
  plt.xlabel('Year')
  plt.ylabel('Temperature (ºC)')
  plt.title('Average Annual Temperature')
  plt.ylim(0, 20)
  plt.legend(loc='upper right')
  plt.gcf().set_size_inches(14, 5)

  # ----------------------------------------------------------------
  # create temperature bar chart for coimbra and porto
  plt.subplot(1, 2, 2)
  plt.bar(yrs_coimbra, precip_coimbra, color='darkorchid', 
          yerr=precip_error_coimbra, align='center', ecolor='black',
          capsize=10, label='Coimbra', width= 0.25)

  plt.bar(yrs_porto +  0.25, precip_porto, color='coral',
          yerr=precip_error_porto, align='center', ecolor='black',
          capsize=10, label='Porto', width= 0.25)

  # set labels, title, legend, and figure size
  plt.xlabel('Year')
  plt.ylabel('Precipitation (mm)')
  plt.title('Total Annual Precipitation')
  plt.legend(loc='upper right')
  plt.gcf().set_size_inches(14, 5)

  # adjust layout for display purposes then show the plot
  plt.tight_layout()
  plt.show()


# ----------------------------------------------------------------
# functions to plot the cover loss attributed to fires
def plot_tree_cover_loss(x_data, y_data, xlabel, ylabel, title, color):
  plt.bar(x_data, y_data, color=color)

  for bar in plt.bar(x_data, y_data, color=color):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}%', ha='center', va='bottom', size=8)

  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.ylim(0, 65)
  plt.title(title)

def loss_percent(dataset,  title1, title2):
  plt.figure(figsize=(20, 6))

  x_var, y_var = 'YEAR', 'PERCENT'
  xlabel, ylabel = 'Year', '% of Tree Cover Loss Due to Fire'

  plt.subplot(1, 2, 1)
  plot_tree_cover_loss(dataset[x_var][22:], dataset[y_var][22:], xlabel, ylabel, title1, color='orchid')

  plt.subplot(1, 2, 2)
  plot_tree_cover_loss(dataset[x_var][:22], dataset[y_var][:22], xlabel, ylabel, title2, color='coral')

  plt.tight_layout()
  plt.show()

# ----------------------------------------------------------------
# function to plot relationship between temperature and precip
# takes two arguments: an array with title, and the confidence level
# outputs 2 charts in 1x2 dimensions
def temp_precip_corr(title_arr, confidence, dataset1, dataset2):

  varx1_temp, vary1_temp = dataset1['TAVG'], dataset1['LOSS_FIRES_HA']
  varx2_temp, vary2_temp = dataset2['TAVG'], dataset2['LOSS_FIRES_HA']
  varx1_precip, vary1_precip = dataset1['PRCP'], dataset1['LOSS_FIRES_HA']
  varx2_precip, vary2_precip = dataset2['PRCP'], dataset2['LOSS_FIRES_HA']

  # calculate pearson correlation coefficient for temp and precipitation
  correlation_coefficient_temp1, p_value_temp1 = pearsonr(varx1_temp, vary1_temp)
  correlation_coefficient_temp2, p_value_temp2 = pearsonr(varx2_temp, vary2_temp)
  correlation_coefficient_precip1, p_value_precip1 = pearsonr(varx1_precip, vary1_precip)
  correlation_coefficient_precip2, p_value_precip2 = pearsonr(varx2_precip, vary2_precip)

  fig, axs = plt.subplots(1, 2, figsize=(12, 5))
  xlabel = ['Temperature (ºC)', 'Precipitation (mm)']
  ylabel = 'Tree Cover Loss (ha)'

  # scatter plot for temperature
  axs[0].scatter(varx1_temp, vary1_temp, label='Coimbra', marker='*', s=70)
  axs[0].scatter(varx2_temp, vary2_temp, label='Porto',  marker='o', s=50)
  axs[0].set_xlabel(xlabel[0])
  axs[0].set_ylabel(ylabel)
  axs[0].set_title(title[0])
  axs[0].legend()
  axs[0].grid(True)

  # scatter plot for precipitation
  axs[1].scatter(varx1_precip, vary1_precip, label='Coimbra', marker='*', s=70)
  axs[1].scatter(varx2_precip, vary2_precip, label='Porto',  marker='o', s=30)
  axs[1].set_xlabel(xlabel[1])
  axs[1].set_ylabel(ylabel)
  axs[1].set_title(title[1])
  axs[1].legend()
  axs[1].grid(True)

  plt.tight_layout()
  plt.show()

  # print orrelation coefficients and p-values
  print("Temperature:")
  print(f"Coimbra: Pearson Correlation Coefficient: {correlation_coefficient_temp1}, p-value: {p_value_temp1}")
  print(f"Porto: Pearson Correlation Coefficient: {correlation_coefficient_temp2}, p-value: {p_value_temp2}")

  print("\nPrecipitation:")
  print(f"Coimbra: Pearson Correlation Coefficient: {correlation_coefficient_precip1}, p-value: {p_value_precip1}")
  print(f"Porto: Pearson Correlation Coefficient: {correlation_coefficient_precip2}, p-value: {p_value_precip2}")

  # check and print statistical significance
  if p_value_temp1 < confidence:
    print("\nThe correlation for temperature in Coimbra is statistically significant.")
  else:
    print("\nThere is no statistically significant correlation for temperature in Coimbra.")

  if p_value_temp2 < confidence:
    print("The correlation for temperature in Porto is statistically significant.")
  else:
    print("There is no statistically significant correlation for temperature in Porto.")

  if p_value_precip1 < confidence:
    print("The correlation for precipitation in Coimbra is statistically significant.")
  else:
    print("There is no statistically significant correlation for precipitation in Coimbra.")

  if p_value_precip2 < confidence:
    print("The correlation for precipitation in Porto is statistically significant.")
  else:
    print("There is no statistically significant correlation for precipitation in Porto.")

