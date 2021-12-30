import pandas as pd
import numpy as np

def concat_rows(df1, df2):
    """this function concatenates the 
    rows of the two DataFrames"""
    row_concat = pd.concat([df1, df2])
    return row_concat

def concat_cols(df1, df2):
    """This function returns concatenated colums of the two dataframes """
    col_concat = pd.concat([df1, df2], axis =1)
    return col_concat

def merge_dfs(df1, df2):
    """This function merges two input dataframes along their columns"""
    merged_df = pd.merge(df1, df2)
    return merged_df

def col_list_sum(df, col_list, weights = None):
    """This is a utility function to calculate the sum  of multiple columns across a DataFrame"""
    col_df = df[col_list]
    if weights is not None:
        col_df = col_df.mutiply(weights)
    return col_df.sum(axis=1)

def get_min_max(data):
    overall_min = data.min()
    overall_max = data.max()
    return overall_min, overall_max

def col_min(data):
    min0 = data.min(axis = 0)
    return min0

def basic_stats(data):
    mean = np.mean(data)
    median = np.median(data)
    var = np.var(data)
    return mean, median, var

def get_sums(data):
  total_sum = np.sum(data)
  col_sum = np.sum(data, axis=0)
  return total_sum, col_sum

def get_cumsum(data):
  row_cumsum = np.cumsum(data, axis=1)
  return row_cumsum


def concat_arrays(data1, data2):
  col_concat = np.concatenate([data1, data2])
  row_concat = np.concatenate([data1, data2], axis=1)
  return col_concat, row_concat

def save_points(save_file):
  points = np.random.uniform(low = -2.5, high = 2.5, size = (100, 2))
  np.save(save_file, points)