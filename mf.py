# install required modules
# import auxiliary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio
import matplotlib.colors as mcolors
import seaborn as sns

# --------------------------------------------------------------------------
# function to create hexbin
def create_hexbin(x_var, y_var, bar_label, x_label, y_label):
  fig, ax = plt.subplots()
  hexbin = ax.hexbin(x, y, gridsize=20, cmap='Blues')
  cbar = plt.colorbar(hexbin, label=bar_label)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
