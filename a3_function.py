# functions for optimized assignment 3
# ------------------------------------
# import auxiliary packages/modules
import os, pandas as pd

# ------------------------------------
# merge multiple csv files with tide
# data into a single one
def merge_csvs(folder_path, csv_filename):
  
  # initialize an empty DataFrame to store merged data
  # then loop through the files in the folder to sort
  # them based on date first then time
  merged_df = pd.DataFrame()
  files_to_merge = []
  for filename in os.listdir(folder_path):
    if filename.startswith("nb_") and filename.endswith(".csv"):
      files_to_merge.append(filename)
  files_to_merge.sort()

  # merge the files into a single one then sort them
  for filename in files_to_merge:
    file_path = os.path.join(folder_path, filename)
    df = pd.read_csv(file_path)
    merged_df = pd.concat([merged_df, df], ignore_index=True)
  
  merged_df["Date"] = pd.to_datetime(merged_df["Date"])
  merged_df["Time (GMT)"] = pd.to_datetime(merged_df["Time (GMT)"], format='%H:%M').dt.time
  merged_df.sort_values(by=["Date", "Time (GMT)"], inplace=True)

  # save the merged dataframe as a csv file then and as a variable, and return it
  merged_file_path = os.path.join(folder_path, csv_filename)
  merged_df.to_csv(merged_file_path, index=False)
  merged_filename = merged_df
  return merged_filename































