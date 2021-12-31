import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import scale, Normalizer, PCA


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


def standardize_data(data):
    scaled_data = scale(data)
    return scaled_data

def normalize_data(data):
  normalizer = Normalizer()
  norm_data = normalizer.fit_transform(data)
  return norm_data

def pca_data(data, n_components):
  pca_obj = PCA(n_components = n_components)
  component_data = pca_obj.fit_transform(data)
  return component_data

def separate_data(component_data, labels, label_names)1:
    separated_data = [] 
    def get_label_info(component_data, labels, class_label, label_names):
        label_name = label_names[class_label]
        label_data = component_data[labels == class_label]
        return (label_name, label_data)
    for class_label in range(len(label_names)):
        separated_data.append(get_label_info(component_data, labels, class_label, label_names))
    return separated_data

def cv_ridge_reg(data, labels, alphas):
    reg = linear_model.RidgeCV(alphas = alphas)
    reg.fit(data, labels)
    return reg

def lasso_reg(data, labels, alpha):
  reg = linear_model.Lasso(alpha=alpha)
  reg.fit(data, labels)
  return reg

def bayes_ridge(data, labels):
    reg = linear_model.BayesianRidge()
    reg.fit(data, labels)
    return reg

def multiclass_lr(data, labels, max_iter):
  reg = linear_model.LogisticRegression(solver = 'lbfgs', max_iter = max_iter, multi_class = 'multinomial')
  reg.fit(data, labels)
  return reg