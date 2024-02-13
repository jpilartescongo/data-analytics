# Import auxiliary functions
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio

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

# function to create animated gif of precipitation across conus
def conus_precip_plot(date_range, data_value, title, label):
  images = []
  colorbar_label = label
  for day in data_range:
    # create temporary image file for each day
    precip_day = data_value[day]
    plt.figure(figsize=(15, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.contourf(conus_precip_lon, conus_precip_lat, precip_day, transform=ccrs.PlateCarree(), cmap='viridis', vmin=0, vmax=180)
    ax.coastlines()
    
    # add gridlines and colorbar 
    ax.gridlines(draw_labels=True)
    plt.colorbar(colorbar_label='Precipitation (mm)')
    plt.title(f'Daily Precipitation - Day {day + 1}')

    # save resulting file
    filename = f'precipitation_day_{day + 1}.png'
    plt.savefig(filename)
    plt.close()
    images.append(filename)

    return images
