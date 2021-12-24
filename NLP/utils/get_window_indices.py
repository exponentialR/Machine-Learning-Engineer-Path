import tensorflow as tf #

def get_window_indices(sequence, target_index, half_window_size):
    left_incl = max(0, target_index - half_window_size)
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return left_incl, right_excl

    