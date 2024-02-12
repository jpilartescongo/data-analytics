# Import auxiliary functions
import numpy as np
import netCDF4 as nc
import pandas as pd
import xarray as xr
import requests
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cartopy.crs as ccrs
import glob, os, time, tarfile
from os.path import exists, join
from IPython.display import clear_output
import datetime as dt
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.tri as Tri
import folium
import cartopy.feature as cfeature
from netCDF4 import Dataset

def plot_gr_map(coords):
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
