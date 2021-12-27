from os import truncate
import tensorflow as tf

from NLP.utils.util_func import create_basic_decoder, get_bi_state_parts, run_decoder 
import tensorflow_addons as tfa

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
        cell_list = [self.make_lstm_cells(dropout_keep_prob, num_units) for i in range(self.num_lstm_layers)]
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
            bi_state_c, bi_state_h = get_bi_state_parts(
                states_fw[i], states_bw[i]
            )
    
    def combine_enc_outputs(self, enc_outputs):
        enc_outputs_fw, enc_outputs_bw = enc_outputs
        return tf.concat([enc_outputs_fw, enc_outputs_bw], -1)

    def create_decoder_cell(self, enc_outputs, input_seq_lens, is_training):
        num_decode_units = self.num_lstm_units
        dec_cell = self.stacked_lstm_cells(is_training, num_decode_units)
        combined_enc_outputs = self.combine_enc_outputs(enc_outputs)
        attention_mechanism = tfa.seq2seq.LuongAttention(num_decode_units, combined_enc_outputs, memory_sequence_length=input_seq_lens)
        dec_cell = tfa.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size = num_decode_units)
        return dec_cell
    
    # Create the sampler for decoding
    def create_decoder_sampler(self, decoder_inputs, is_training, batch_size):
        if is_training:
            dec_embeddings, dec_seq_lens = self.get_embeddings(decoder_inputs, 'decoder_emb')
            sampler = tfa.seq2seq.sampler.TrainingSampler()
            pass
        else:
            embedding_matrix = tf.keras.layers.Embedding(self.vocab_size, int(self.vocab_size/2))
            sampler = tfa.seq2seq.sampler.GreedyEmbeddingSampler(embedding_matrix)
            dec_seq_lens = None
        return sampler, dec_seq_lens
    
    def Decoder(self, enc_outputs, input_seq_lens, final_state, batch_size,
        sampler, dec_seq_lens , is_training):
        dec_cell = self.create_decoder_cell(enc_outputs, input_seq_lens, is_training)
        projection_layer = tf.keras.layers.Dense(self.extended_vocab_size)
        batch_s = tf.constant(batch_size)

        
        initial_state = dec_cell.get_initial_state(enc_outputs[0],batch_size=batch_s , dtype = tf.float32)

        decoder = tfa.seq2seq.BasicDecoder(
            dec_cell, sampler,
            output_layer=projection_layer
            )
        inputs = enc_outputs[0]
        output = run_decoder(decoder,inputs , initial_state , input_seq_lens , is_training , dec_seq_lens)
        return output
    
    def calculate_loss(self, logits, dec_seq_lens, decoder_outputs, batch_size):
        binary_sequences = tf.compat.v1.sequence_mask(dec_seq_lens, dtype=tf.float32)
        batch_loss = tf.compat.v1.nn.sparse_softmax_cross_entropy_with_logits(
            labels=decoder_outputs, logits=logits)
        unpadded_loss = batch_loss * binary_sequences
        per_seq_loss = tf.math.reduce_sum(unpadded_loss) / batch_size
        return per_seq_loss