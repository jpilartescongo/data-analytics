# install required modules
# import auxiliary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import matplotlib.colors as mcolors
import seaborn as sns

# --------------------------------------------------------------------------
# function to group features according to location
def indiv_station(dataset, city_name):
  feature = dataset[dataset['NAME'].str.startswith(city_name)]
  return feature

# --------------------------------------------------------------------------
# function to create hexbin
def create_hexbin(x_var, y_var, bar_label, x_label, y_label):
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

# def generate_density_plot(data, ax, x_label, title):
#   sns.distplot(data, ax=ax)
#   ax.set_xlabel(x_label)
#   ax.set_title(title)


# function to create and display histogram
def create_hist2(dataset_arr, limit_arr, mean_label_position, plot_color, title, histx_label_arr, histy_label_arr, var_array, unit_array):
    fig, axs = plt.subplots(1,2, figsize=(13,4))
    xlim, ylim = (limit_arr[0], limit_arr[1]), (limit_arr[2], limit_arr[3])
    label_x, label_y = mean_label_position[0], mean_label_position[1]

    mean1 = dataset_arr[0].mean()
    dataset_arr[0].plot(kind='hist', bins=20, ax=axs[0], color=plot_color, edgecolor='black', linewidth=0.5)
    axs[0].set_title(title_arr[0])
    axs[0].spines[['top', 'right']].set_visible(True)
    axs[0].set_xlabel(histx_label_arr[0])
    axs[0].set_ylabel(histy_label_arr[0])
    axs[0].axvline(mean1, color='black', linestyle='dashed', linewidth=1)
    axs[0].set_xlim(hist_x_lim)
    axs[0].set_ylim(hist_y_lim)
    axs[0].text(mean_label_x, mean_label_y, f'{var_array[0]} (Mean): {mean2:.2f} {unit_array[0]}'.format(var_array[0], unit_array[0]), ha='center')  
        
    mean2 = dataset_arr[1].mean()
    dataset_arr[1].plot(kind='hist', bins=20, ax=axs[0], color=plot_color, edgecolor='black', linewidth=0.5)
    axs[1].set_title(title_arr[1])
    axs[1].spines[['top', 'right']].set_visible(True)
    axs[1].set_xlabel(histx_label_arr[1])
    axs[1].set_ylabel(histy_label_arr[1])
    axs[1].axvline(mean2, color='black', linestyle='dashed', linewidth=1)
    axs[1].set_xlim(hist_x_lim)
    axs[1].set_ylim(hist_y_lim)
    axs[1].text(mean_label_x, mean_label_y, f'{var_array[1]} (Mean): {mean2:.2f} {unit_array[1]}'.format(var_array[1], unit_array[1]), ha='center')
    
    return fig
