import tensorflow as tf 


class SqueezeNetModel(object):

    def __init__(self, original_dim, resize_dim, output_size):
        self.original_dim = original_dim
        self.resize_dim = resize_dim
        self.output_size = output_size
    
    #Random Crop and Flip 

    def random_crop_flip(self, float_image):
        crop_image = tf.compat.v1.random_crop(float_image, [self.resize_dim, self.resize_dim, 3])
        updated_image = tf.image.random_flip_left_right(crop_image)
        return updated_image
    
    def image_preprocessing(self, data, is_training):
        reshaped_image = tf.reshape(data, [3, self.original_dim, self.original_dim])
        transposed_image = tf.transpose(reshaped_image, [1, 2, 0])
        float_image = tf.cast(transposed_image, tf.float32)
        if is_training:
            updated_image = self.random_crop_flip(float_image)
        else:
            updated_image = tf.image.resize_image_with_crop_or_pad(float_image, self.resize_dim, self.resize_dim)
        standardized_image = tf.image.per_image_standardization(updated_image)
        return standardized_image

    
    def custom_conv2d(self, inputs, filters, kernel_size, name):
        return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        name=name)(inputs)

    # SqueezeNet fire module
    def fire_module(self, inputs, squeeze_depth, expand_depth, name):
        with tf.compat.v1.variable_scope(name):
            squeezed_inputs = self.custom_conv2d(
                inputs,
                squeeze_depth,
                [1, 1],
                'squeeze')
            expand1x1 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [1, 1],
                'expand1x1')
            expand3x3 = self.custom_conv2d(
                squeezed_inputs,
                expand_depth,
                [3, 3],
                'expand3x3')
            return tf.concat([expand1x1, expand3x3], axis=-1)
    

    def custom_max_pooling2d(self, inputs, name):
        return tf.keras.layers.MaxPool2D(
        pool_size=[2, 2],
        strides=2,
        name=name)(inputs)

    # Model Layers
    # inputs: [batch_size, resize_dim, resize_dim, 3]
    def model_layers(self, inputs, is_training):
        # CODE HERE
        conv1 = self.custom_conv2d(
            inputs,
            64,
            [3, 3],
            'conv1')
        pool1 = self.custom_max_pooling2d(
            conv1,
            'pool1')
        fire_params1 = [
            (32, 64, 'fire1'),
            (32, 64, 'fire2')
        ]
        multi_fire1 = self.multi_fire_module(
            pool1,
            fire_params1)
        pool2 = self.custom_max_pooling2d(
            multi_fire1,
            'pool2')

        
    def multi_fire_module(self, layer, params_list):

        # CODE HERE
        for params in params_list:
            layer = self.fire_module(
                layer,
                params[0],
                params[1],
                params[2])
        return layer