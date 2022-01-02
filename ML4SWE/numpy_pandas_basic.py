import pandas as pd
import numpy as np
from sklearn import linear_model, tree, metrics
from sklearn.preprocessing import scale, Normalizer, PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import MiniBatchKMeans, KMeans, FeatureAgglomeration

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


def dataset_splitter(data, labels, test_size=0.25):
  split_dataset = train_test_split(data, labels, test_size = test_size)
  train_set = (split_dataset[0], split_dataset[2])
  test_set = (split_dataset[1], split_dataset[3])
  return train_set, test_set


def cv_decision_tree(is_clf, data, labels,
                     max_depth, cv):
  if is_clf:
    d_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
  else:
    d_tree = tree.DecisionTreeRegressor(max_depth=max_depth)
  scores = cross_val_score(d_tree, data, labels, cv=cv)
  return scores

def evaluate_regression_model(train_data, train_labels, test_data, test_labels):
    reg = tree.DecisionTreeRegressor()
    Error_Predict = {}
    # predefined train and test sets
    reg.fit(train_data, train_labels)
    predictions = reg.predict(test_data)
    r2 = metrics.r2_score(test_labels, predictions)
    Error_Predict['R-Squared'] = r2
    print('R2: {}\n'.format(r2))
    mse = metrics.mean_squared_error(test_labels, predictions)
    Error_Predict['Mean Squared Error'] = mse
    print('MSE: {}\n'.format(mse))
    mae = metrics.mean_absolute_error(test_labels, predictions)
    Error_Predict['Mean Absolute Error'] = mae
    print('MAE: {}\n'.format(mae))
    return Error_Predict

def evaluate_classification_model(train_data, train_labels, test_data, test_labels):
    clf = tree.DecisionTreeClassifier()
    # predefined train and test sets
    clf.fit(train_data, train_labels)
    predictions = clf.predict(test_data)
    acc = metrics.accuracy_score(test_labels, predictions)
    print('Accuracy: {}\n'.format(acc))


def kmeans_clustering(data, n_clusters, batch_size):
  if batch_size is None:
    kmeans = KMeans(n_clusters=n_clusters)
  else:
    kmeans = MiniBatchKMeans(n_clusters=n_clusters,
                             batch_size=batch_size)
  kmeans.fit(data)
  return kmeans


def feature_Agglomerate_PCA(data):
    agg = FeatureAgglomeration(n_clusters=2)
    new_data = agg.fit_transform(data)
    print('New shape: {}\n'.format(new_data.shape))
    print('First 10:\n{}\n'.format(repr(new_data[:10])))
    return new_data