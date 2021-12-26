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
    
    def make_lstm_cells(self, dropout_kep_prob, num_units):
        cell = tf.keras.layers(num_units, dropout = dropout_kep_prob)
        return cell 

    def get_embeddings(self, sequences, scope_name):
        with tf.compat.v1.variable_scope(scope_name, reuse=tf.compat.v1.AUTO_REUSE):
            cat_column = tf.compat.v1.feature_column.categorical_column_with_identity('sequences', self.extended_vocab_size)
            embed_size = int(self.extended_vocab_size**0.25)
            embedding_column = tf.compat.v1.feature_column.embedding_column(cat_column, embed_size)
            seq_dict = {'sequences': sequences}
            embeddings = tf.compat.v1.feature_column.input_layer(seq_dict, [embedding_column])
            sequence_lengths = tf.compat.v1.placeholder('int64', shape = (None,), name = scope_name+"/sinput_layer/sequence_length")
            return embeddings, tf.cast(sequence_lengths, tf.int32)
    
    def stacked_lstm_cells(self, is_training, num_units):
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell_list = self[self.make_lstm_cells(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
        cell = tf.keras.layers.StackedRNNCells(cell_list)
        return cell
    
    def encoder(self, encoder_inputs, is_training):
        input_embeddings, input_seq_lens = self.get_embeddings(encoder_inputs, 'encoder_emb')
        cell = self.stacked_lstm_cells(is_training, self.num_lstm_units)


        rnn = tf.keras.layers.RNN(cell, return_sequences = True, return_state = True, go_backwards = True, dtype = tf.float32)
        Bi_rnn = tf.keras.layers.Biderectional(rnn, merge_mode = 'concat')
        input_embeddings = tf.reshape(input_embeddings, [-1,-1,2])
        outputs = Bi_rnn(input_embeddings)

        states_fw =  [ outputs[i]  for i in range(1,self.num_lstm_layers+1)] 
        states_bw =  [ outputs[i]  for i in range(self.num_lstm_layers+1,len(outputs))]

        for i in range(self.num_lstm_layers):
            bi_state_c, bi_state_h = ref_get_bi_state_parts(
                states_fw[i], states_bw[i]
            )
