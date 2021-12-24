import tensorflow as tf

def lstm_make_predictions(shape):  
    """ This makes prediction on the trained model
    shape in this case is a tuple and can be in the format e.g. (None, 5, 100)"""
    probabilities = tf.compat.v1.placeholder(tf.float32, shape=shape)
    word_preds = tf.compat.v1.argmax(probabilities, axis=-1)
    return word_preds