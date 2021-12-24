import tensorflow as tf #

def get_window_indices(sequence, target_index, half_window_size):
    left_incl = max(0, target_index - half_window_size)
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return left_incl, right_excl


def get_target_and_size(sequence, target_index, window_size):
    half_window_size = window_size//2
    target_word = sequence[target_index]
    return target_word, half_window_size


def get_initializer(embeddin_dim, vocab_size):
    initial_bounds = 0.5/embedding_dim
    initializer = tf.random.uniform((vocab_size, embeddin_dim), minval=initial_bounds, maxval=initial_bounds)
    return initializer


def truncate_sequences(sequence, max_length):
    input_sequence = sequence[:max_length-1]
    target_sequence = sequence[1:max_length]
    return input_sequence, target_sequence

def pad_sequences(sequence, max_length):
    padding_amount = max_length - len(sequence)
    padding = [0 for i in range(padding_amount)]
    input_sequence = sequence[:-1] + padding
    target_sequence = sequence[1:] + padding
    return input_sequence, target_sequence