# import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

# def init_inputs(input_size):
#     """This function is returns tf placeholder for model's input data"""
#     inputs = tf.compat.v1.placeholder(tf.float32, shape = (None, input), name = 'inputs')
#     return inputs 

# def init_labels(output_size):
#     """This is a placeholder for labels"""
#     labels = tf.compat.v1.placeholder(tf.int32, shape = (None, output_size), name = 'labels')
#     return labels

# def model_layers(inputs, output_size):
#   logits = tf.keras.layers.Dense(output_size,
#                            name='logits')(inputs)
#   return logits


def convert_dataframe(first_array, second_array):
    keypoint_array = {'first_array': first_array, 'second_array': second_array}
    return pd.DataFrame.from_dict(keypoint_array)
     

def plot_scatter_plots(x, y):
    
    pl.plot(x, y, 'ro', alpha = 0.5)
    # pl.plot(array2[:,0], array2[:, 1], 'ro', alpha = 0.5)
    # for i in range(array1.shape[0]):
    #     pl.text(array1[i, 0], array1[i, 1], str(i))
    pl.show()


# plot_scatter_plots(100, 200)
# y = np.load(r"E:\keyData\intentiondata_13\intention_1\time_since_action_start.npy")
x = np.load(r"C:\Users\40311630\Downloads\intentiondata_19\intentions\0\keypoint_1.npy")
y = np.load(r'C:\Users\40311630\Downloads\intentiondata_19\intentions\0\keypoint_15.npy')
# print(y)
# gmail = convert_dataframe(x, y)
# print(gmail)
# plot_scatter_plots(x, y)
# print(y.shape)
# from matplotlib import pyplot as plt

# import numpy as np
# N = 100
# matrix = np.random.rand(N,2)

# plt.plot(matrix[:,0],matrix[:,1], 'ro', alpha = 0.5)
# for i in range(matrix.shape[0]):
#     plt.text(matrix[i,0], matrix[i,1], str(i))

# plt.show()
# actions = ['intention', 'no_intention']
# sequences, labels = [], []
# for action in actions:
#     for sequence in range(30):
#         window =[]
#         res = np.load(os.path.join)
import os
# def getListOfFiles(dirName):
#     # create a list of file and sub directories 
#     # names in the given directory 
#     listOfFile = os.listdir(dirName)
#     allFiles = list()
#     # Iterate over all the entries
#     for entry in listOfFile:
#         # Create full path
#         fullPath = os.path.join(dirName, entry)
#         # If entry is a directory then get the list of files in this directory 
#         if os.path.isdir(fullPath):
#             allFiles = allFiles + getListOfFiles(fullPath)
#         else:
#             allFiles.append(fullPath)
                
#     return allFiles       

dir_name = r'C:\Users\40311630\Downloads\intentiondata_21'
# listOfFiles = getListOfFiles(dir_name)
# listdr = []

# # Print the files
# for elem in listOfFiles:
#     listdr.append(elem)
# print(len(listdr))
actions = ['intentions', 'notintentions']
label_map = {label:num for num, label in enumerate(actions)}
# np_load = np.load
# np.load = lambda *a, **k: np_load(*a, allow_pickle=True, **k)
print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(5):
        window =[]
        for frame_num in range(1, 41):
            
            res_eyes = np.load(os.path.join(dir_name, action, str(sequence), 'keypoint_{}.npy'.format(frame_num)), allow_pickle=True)
            # print(res_eyes)
        sequences.append(window)
        labels.append(label_map[action])
    # print(np.array(sequences).shape)
    # print(np.array(labels).shape)
X = np.array(sequences, dtype=object)  
print(np.array(sequences, dtype=object).shape)
print(X.shape[1])
eyes = sequences[1:10000]
print(eyes)
# # print(sequences)
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split

# Y = to_categorical(labels).astype(int)
# print(Y)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)
# print(X_train.shape)

# intention_path = r'C:\Users\40311630\Downloads\intentiondata_19\intentions'
# no_intent = r'C:\Users\40311630\Downloads\intentiondata_19\nointentions'

# for #
# x = np.array([5,2,4,6,7])
# print(x)
# y = np.array([11, 22, 9, 78,10])
# w = np.concatenate([x, y]).flatten()
# print(w[:2])