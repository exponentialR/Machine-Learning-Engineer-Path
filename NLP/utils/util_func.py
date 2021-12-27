import tensorflow as tf #
import tensorflow_addons as tfa

def get_window_indices(sequence, target_index, half_window_size):
    left_incl = max(0, target_index - half_window_size)
    right_excl = min(len(sequence), target_index + half_window_size + 1)
    return left_incl, right_excl


def get_target_and_size(sequence, target_index, window_size):
    half_window_size = window_size//2
    target_word = sequence[target_index]
    return target_word, half_window_size


def get_initializer(embeddin_dim, vocab_size):
    initial_bounds = 0.5/embeddin_dim
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

def get_bi_state_parts(state_fw, state_bw):
    bi_state_c = tf.concat([state_fw[0], state_bw[0]], -1)
    bi_state_h = tf.concat([state_fw[1], state_bw[1]], -1)
    return bi_state_c, bi_state_h

def create_basic_decoder(enc_outputs , extended_vocab_size, batch_size, final_state, dec_cell, sampler):
    projection_layer = tf.keras.layers.Dense(extended_vocab_size)
    batch_size = tf.constant(batch_size)
    initial_state = dec_cell.get_initial_state(enc_outputs[0],batch_size=batch_size , dtype = tf.float32)
    decoder = tfa.seq2seq.BasicDecoder(
            dec_cell, sampler,
            output_layer=projection_layer
            )
    return decoder

def run_decoder(decoder,inputs , initial_state , input_seq_lens , isTraining ,dec_seq_lens):
    dec_outputs, _, _ =decoder(inputs , initial_state=initial_state , sequence_length=input_seq_lens , training=isTraining) 
    if isTraining:
        logits = dec_outputs.rnn_output
        return logits, dec_seq_lens
    return dec_outputs.sample_id