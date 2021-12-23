"""Basic Decoding"""

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import make_one_shot_iterator 
def decode_image(filename, image_type, resize_shape, channels =0):
    value = tf.io.read_file(filename)
    if image_type == 'png':
        decoded_image = tf.io.decode_png(value, channels = channels)
    elif image_type == 'jpeg':
        decoded_image = tf.io.decode_jpeg(value, channels = channels)
    else: 
        decoded_image = tf.io.decode_image(value, channels = channels)
    
    if resize_shape is not None and image_type in ['png', 'jpeg']:
        decoded_image = tf.image.resize(decoded_image, resize_shape)
    return decoded_image

#Creating Dataset using tf.data.Dataset.from_tensor_slices

def get_dataset(image_paths, image_type, resize_shape, channels):
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor)
    def _map_func(filename):
        return decode_image(filename, image_type, resize_shape, channels = channels)
    return dataset.map(_map_func)


#Setting up iterator to extract data from a pixel array dataset
def get_image_data(image_paths, image_type = None, resize_shape = None, channels = 0):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels=channels)
    iterator = tf.compat.v1.data,make_one_shot_iterator(dataset)
    next_image = iterator.get_next
    image_data_list = []
    with tf.compat.v1.Session() as sess:
        for i in range(len(image_paths)):
            image_data = sess.run(next_image)
            image_data_list.append(image_data)
        return image_data_list


with tf.compat.v1.Session() as sess:
    decoded_image = decode_image('WIN_20211222_14_49_55_Pro.jpg', 'jpg', (3,2))
