# Import auxiliary functions
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import matplotlib.colors as mcolors

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
# function to create subplots in a 3x2 grid
def point_data_subplots(dataset, x_variable, list_of_variables, plot_title):
  list = list_of_variables
  fig, axs = plt.subplots(3, 2, figsize=(5,5))
  fig.suptitle(plot_title)

  # loop through a list of variables to be ploted
  attrs = []
  for i in list: 
    attrs.append(i) 

  # plots for first location (point data)
  # variables: wind, precip, min/max temp, avg temp
  dataset.plot(x=x_variable, y=attrs[0], ax=axs[0,0])
  dataset.plot(x=x_variable, y=attrs[1], ax=axs[0,1])
  dataset.plot(x=x_variable, y=attrs[2], ax=axs[1,0])
  dataset.plot(x=x_variable, y=attrs[3], ax=axs[1,1])
  dataset.plot(x=x_variable, y=attrs[4], ax=axs[2,0])
  dataset.plot(x=x_variable, y=attrs[5], ax=axs[2,1])

  # display plots
  return plt.show()
