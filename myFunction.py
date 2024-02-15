# Import auxiliary functions
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import matplotlib.colors as mcolors
import seaborn as sns

# ------------------------------------------
# clear output of colab cells
from IPython.display import clear_output
clear_output()


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

def generate_density_plot(data, ax, x_label, title):
  sns.distplot(data, ax=ax)
  ax.set_xlabel(x_label)
  ax.set_title(title)
