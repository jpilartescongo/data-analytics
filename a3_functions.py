# functions for assignment 3 (data analytics)
# ----------------------------------------------
# import auxiliary packages/modules
import os, pandas as pd

# ----------------------------------------------
# merge multiple csv files into a single one
def merge_csvs(folder_path, csv_filename, init_chars):
  # initialize an empty DataFrame to store merged data
  # then loop through the files in the folder to sort
  # them based on date first then time
  merged_df = pd.DataFrame()
  files_to_merge = []
  for filename in os.listdir(folder_path):
    if filename.startswith(init_chars) and filename.endswith(".csv"):
      files_to_merge.append(filename)
  files_to_merge.sort()

  # merge the files into a single one then sort them
  for filename in files_to_merge:
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, df], ignore_index=True)
  
  # convert date and time columns to datetime then resample to hour frequency
  merged_df['DateTime'] = pd.to_datetime(merged_df['Date'] + ' ' + merged_df['Time (GMT)'])
  merged_df.drop(['Date', 'Time (GMT)'], axis=1, inplace=True)
  merged_df = merged_df.set_index('DateTime').resample('H').mean(numeric_only=True).reset_index()

  # save the merged dataframe as a csv file and return merged dataframe
  merged_file_path = os.path.join(folder_path, csv_filename)
  merged_df.to_csv(merged_file_path, index=False)
  return merged_df






















