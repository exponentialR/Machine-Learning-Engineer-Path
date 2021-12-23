import tensorflow as tf 

class MNISTModel(object):

    def __init__(self, input_dim, output_size):
        self.input_dim = input_dim
        self.output_size = output_size

        """In order to use the data with convolutional neural network, 
        It has to be in the NHWC Format 
        N ==> Number of image data samples (batch_size)
        H ==> Height of each image
        W ==> Width of each image 
        C ==> Channels per image 
        
        The number of data samples may vary and can be specified later, 
        The height and width of the image data is from the dataset self.input_dim
        While the number of channels is 1 (since images are in greyscales)"""
    def model_layers(self, inputs, is_training):
        reshaped_inputs = tf.reshape(inputs, [-1, self.input_dim, self.input_dim, 1])