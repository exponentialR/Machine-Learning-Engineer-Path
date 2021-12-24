import tensorflow as tf 

def get_target_and_size(sequence, target_index, window_size):
    half_window_size = window_size//2
    target_word = sequence[target_index]
    return target_word, half_window_size