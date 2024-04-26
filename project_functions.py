# ancillary libraries/modules for final project
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os, pandas as pd, sys, np

# function that creates histogram plots for a dataframe
def geotag_histograms(dataframe, y_lim):
    # Create a figure with 3 subplots in 1 row
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    axs[0].hist(dataframe['lat'], bins=30, color='blue', edgecolor='black')
    axs[0].set_title('Latitude')
    axs[0].set_xlabel('Latitude')
    axs[0].set_ylabel('Frequency')
    axs[0].set_ylim(y_lim[0])

    axs[1].hist(dataframe['lon'], bins=30, color='green', edgecolor='black')
    axs[1].set_title('Longitude')
    axs[1].set_xlabel('Longitude')
    axs[1].set_ylabel('Frequency')
    axs[1].set_ylim(y_lim[0])

    axs[2].hist(dataframe['alt'], bins=30, color='red', edgecolor='black')
    axs[2].set_title('Altitude')
    axs[2].set_xlabel('Altitude')
    axs[2].set_ylabel('Frequency')
    axs[2].set_ylim(y_lim[1])
    
    # adjust the plot and display results
    plt.tight_layout()
    plt.show()
