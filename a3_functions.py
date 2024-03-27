# functions for assignment 3 (data analytics)
# ----------------------------------------------
# import auxiliary packages/modules
import os, pandas as pd

# ----------------------------------------------
# get the count of nans for various variables
def get_count_of_nans(df, vars):
  nan_counts = df[vars].isna().sum()
  nan_counts_df = pd.DataFrame(nan_counts, columns=['Count'])
  return nan_counts_df
