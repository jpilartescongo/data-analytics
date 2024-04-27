# ancillary libraries/modules for the final project
import numpy as np
import scipy.stats as stats
import os, pandas as pd, sys
import matplotlib.pyplot as plt

#---------------------------------------------------------------------------
# function that creates histogram plots for an input dataframe with geotags
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

#---------------------------------------------------------------------------
# function that outputs the equations of the regression model for predicting
def print_equations(coefficients, intercepts, features, targets):
    for i, target in enumerate(targets):
        terms = [f"{coefficients[i][j]:.4f}*{features[j]}" for j in range(len(features))]
        equation = " + ".join(terms)
        print(f"{target} = {intercepts[i]:.4f} + {equation}")

#---------------------------------------------------------------------------
# function that creates a pairwise correlation between features for a given
# dataframe; it takes two inputs: the dataframe variable name and the names
# of features to plot (array of strings that represent the variables/fields)
def create_pairwise_correlation(dataframe, attributes):
    correlation_matrix = dataframe[attributes].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=.3)
    
    plt.title('Pairwise Correlation Matrix (Uncorrected Dataframe)', fontsize=13)
    plt.tight_layout()
    plt.show()
