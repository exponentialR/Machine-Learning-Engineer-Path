from os import truncate
import tensorflow as tf 

class Seq2SeqModel(object):
    def __init__(self, vocab_size, num_lstm_layers, num_lstm_units):
        self.vocab_size = vocab_size
        self.extended_vocab_size = vocab_size + 2
        self.num_lstm_layers = num_lstm_layers
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = vocab_size)

    
    def make_training_tuple(self, input_sequence, output_sequence):
        truncate_front = output_sequence[1:]
        truncate_back = output_sequence[:-1]
        sos_token = [self.vocab_size]
        eos_token = [self.vocab_size + 1]
        input_sequence = sos_token + input_sequence + eos_token
        ground_truth = sos_token + truncate_back
        final_sequence = truncate_front + eos_token
        return input_sequence, ground_truth, final_sequence