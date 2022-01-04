import tensorflow as tf

def init_inputs(input_size):
    """This function is returns tf placeholder for model's input data"""
    inputs = tf.compat.v1.placeholder(tf.float32, shape = (None, input), name = 'inputs')
    return inputs 

def init_labels(output_size):
    """This is a placeholder for labels"""
    labels = tf.compat.v1.placeholder(tf.int32, shape = (None, output_size), name = 'labels')
    return labels

def model_layers(inputs, output_size):
  logits = tf.keras.layers.Dense(output_size,
                           name='logits')(inputs)
  return logits
