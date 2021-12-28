import pandas as pd

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
    