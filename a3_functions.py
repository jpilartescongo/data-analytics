# functions for assignment 3 (data analytics)
# ----------------------------------------------
# import auxiliary packages/modules
import os, pandas as pd

# ----------------------------------------------
# get the count of nans for various variables
def get_count_of_nans(dataset, variables):
  nan_count = dataset[variables].isna().sum()
  #nan_count_df = pd.DataFrame(nan_count, columns=['Count:'])
  return nan_count_df
