# ancillary libraries/modules for the final project
import numpy as np
import seaborn as sns
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
    
#---------------------------------------------------------------------------
# function to plot altitude differences (actual vs predicted) for wingtra
# dataset; for simplicity, results filter only the 30 geotag corrections
def plot_alt_differences(predicted_df, actual_df, title):
    plt.figure(figsize=(20, 10))
    x = np.arange(30)
    plt.bar(x, predicted_df['alt_pred'][:30], label='Predicted', color='coral', width=0.3)
    plt.bar(x + 0.4, actual_df['alt'][:30], label='Actual', color='darkorchid', width=0.3)
    plt.legend()
    plt.title(title, fontsize=20)
    plt.xlabel('Corresponding UAS Image (#)', fontsize=15)
    plt.ylabel('UAS Altitude (m)', fontsize=15)
    plt.show()

#---------------------------------------------------------------------------
# function that creates actual versus predicted lat, lon, and alt values
def plot_lat_lon_alt_differences(aligned_data, fields):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    pred_labels = [f'{field}_pred' for field in fields]
    actual_labels = fields
    titles = ['Latitude', 'Longitude', 'Altitude']
    
    for i, (pred, actual, title) in enumerate(zip(pred_labels, actual_labels, titles)):
        ax[i].plot(aligned_data[pred], label='Predicted')
        ax[i].plot(aligned_data[actual], label='Actual')
        ax[i].set_title(title)
        ax[i].set_xlabel('Corresponding UAS Image (#)')
        ax[i].set_ylabel('Value')
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.show()
    
#---------------------------------------------------------------------------
# function that creates actual versus predicted horizontal/vertical accuracy
def plot_acc_differences(dataframe, y_lim):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x_label, y_label = 'Corresponding UAS Image (#)', 'Accuracy (m)'
    legend_label = ['Predicted', 'Actual']
    
    # plot for horizontal accuracy
    ax[0].plot(dataframe['h_acc_pred'], label=legend_label[0])
    ax[0].plot(dataframe['h_acc'], label=legend_label[1])
    ax[0].set_title('Horizontal Accuracy')
    ax[0].set_ylim(y_lim)
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel(y_label)
    
    # plot for vertical accuracy
    ax[1].plot(dataframe['v_acc_pred'], label=legend_label[0])
    ax[1].plot(dataframe['v_acc'], label=legend_label[1])
    ax[1].set_title('Vertical Accuracy')
    ax[1].set_ylim(y_lim)
    ax[1].set_xlabel(x_label)
    ax[1].set_ylabel(y_label)
    
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.tight_layout()
    plt.show()
    
#---------------------------------------------------------------------------
# supplementary functions that address prior issues with modeling wingtra
# and dji all within the same notebook; this is not the most efficient way
# of addressing the problem but it is a temporary solution; these functions
# predict the latitude, longitude, altitude, and horizontal and vertical
# accuracies of a new dataset using the patterns learned from the model
def load_csv_file(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data.columns = ['img', 'lat', 'lon', 'alt', 'omega', 'phi', 'kappa', 'h_acc', 'v_acc']
    return data

def dji_linear_regression_prediction(csv_file_path, model):
    data = load_csv_file(csv_file_path)
    features = data[['lat', 'lon', 'alt', 'h_acc', 'v_acc']]
    features.columns = ['lat_unc', 'lon_unc', 'alt_unc', 'h_acc_unc', 'v_acc_unc']
    predictions = model.predict(features)
    data['lat_pred'] = predictions[:, 0]
    data['lon_pred'] = predictions[:, 1]
    data['alt_pred'] = predictions[:, 2]
    data['h_acc_pred'] = predictions[:, 3]
    data['v_acc_pred'] = predictions[:, 4]
    data.drop(['lat', 'lon', 'alt', 'h_acc', 'v_acc'], axis=1, inplace=True)
    predicted_data = data[['img', 'lat_pred', 'lon_pred', 'alt_pred', 'h_acc_pred',
                           'v_acc_pred', 'omega', 'phi', 'kappa']]
    return predicted_data

#---------------------------------------------------------------------------
# function that runs performance metrics on dji predictions versus ground
# truth data and outputs results (lat, lon, alt, h_acc, and v_acc)
def get_prediction_metrics(linear_prediction_df, ground_truth_df, aligned_data):
    mse2 = np.mean((linear_prediction_df2 - ground_truth_df2)**2)
    rmse2 = np.sqrt(mse2)
    mae2 = np.mean(np.abs(linear_prediction_df2 - ground_truth_df2))
    return rmse2
    
    lat_rmse2 = get_comparison_metrics2(aligned_data['lat_pred'], aligned_data['lat'])
    lon_rmse2 = get_comparison_metrics2(aligned_data['lon_pred'], aligned_data['lon'])
    alt_rmse2 = get_comparison_metrics2(aligned_data['alt_pred'], aligned_data['alt'])
    h_acc_rmse2 = get_comparison_metrics2(aligned_data['h_acc_pred'], aligned_data['h_acc'])
    v_acc_rmse2 = get_comparison_metrics2(aligned_data['v_acc_pred'], aligned_data['v_acc'])
    
    print('The RMSE values for latitude, longitude, and altitude are as follows:')
    print(f'Latitude RMSE (DJI): {lat_rmse2}')
    print(f'Longitude RMSE (DJI): {lon_rmse2}')
    print(f'Altitude RMSE (DJI): {alt_rmse2}')
    print(f'Horizontal Accuracy RMSE (DJI): {h_acc_rmse2}')
    print(f'Vertical Accuracy (DJI): {v_acc_rmse2}')

