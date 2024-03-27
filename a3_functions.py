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
  
  merged_df["Date"] = pd.to_datetime(merged_df["Date"])
  merged_df["Time (GMT)"] = pd.to_datetime(merged_df["Time (GMT)"], format='%H:%M').dt.time
  merged_df.sort_values(by=["Date", "Time (GMT)"], inplace=True)

  # save the merged dataframe as a csv file then and as a variable, and return it
  merged_file_path = os.path.join(folder_path, csv_filename)
  merged_df.to_csv(merged_file_path, index=False)
  merged_filename = merged_df
  return merged_filename

# ----------------------------------------------
# extract month from a csv and add to new column

def extract_month(dataframe, date_column):
  # set month names for later usage
  month_mapping = {
      1: 'January',
      2: 'February',
      3: 'March',
      4: 'April',
      5: 'May',
      6: 'June',
      7: 'July',
      8: 'August',
      9: 'September',
      10: 'October',
      11: 'November',
      12: 'December'
  }

  # create a new column to store month names as a string
  dataframe['Month_Name'] = dataframe[date_column].dt.month.map(month_mapping)
  return dataframe



























