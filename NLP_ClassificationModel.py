import tensorflow as tf 
#tf.compat.v1.disable_eager_execution()


class ClassificationModel(object):

    def __init__(self, vocab_size, max_length, num_lstm_units):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_lstm_units = num_lstm_units
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = self.vocab_size)

    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences 

    def make_training_pairs(self, texts, labels):
        sequences = self.tokenize_text_corpus(texts)
        for i in range(len(sequences)):
            sequence = sequences[i]
            if len(sequence)>self.max_length:
                sequences[i] = sequence[:self.max_length]
                training_pairs = list(zip(sequences, labels))
                return training_pairs
    
    def make_lstm_cell(self, dropout_keep_prob):
        cell = tf.keras.layers.LSTMCell(self.num_lstm_units, dropout = dropout_keep_prob)
        return cell
    
    def get_input_embeddings(self, input_sequences):
        input_col = tf.compat.v1.feature_column.categorical_column_with_identity('inputs', self.vocab_size)
        embed_size = int(self.vocab_size**0.25)
        embed_col = tf.compat.v1.feature_column.embedding_column(input_col, embed_size)
        input_dict = {'inputs': input_sequences}
        input_embeddings = tf.compat.v1.feature_column.input_layer(input_dict, [embed_col])
        sequence_lengths = tf.compat.v1.placeholder('int64', shape = (None,), name = 'input_layer/input_embedding/sequence_length')
        return input_embeddings, sequence_lengths
    
    def run_bilstm(self, input_sequences, is_training):
        input_embeddings, sequence_lengths = self.get_input_embeddings(input_sequences)
        dropout_keep_prob = 0.5 if is_training else 1.0
        cell = self.make_lstm_cell(dropout_keep_prob)
        rnn = tf.keras.layers.RNN(cell, return_sequences = True, go_backward=True, return_state = True)
        Bi_rnn = tf.keras.layers.Bidirectional(rnn, merge_mode = None)
        input_embeddings = tf.compat.v1.placeholder(tf.float32, shape=(None, 10, 12))
        outputs = Bi_rnn(input_embeddings)
        return outputs, sequence_lengths
    
    def get_gather_indices(self, batch_size, sequence_lengths):
        row_indices = tf.rannge(batch_size)
        final_indexes = tf.cast(sequence_lengths - 1, tf.int32)
        return tf.transpose([row_indices, final_indexes])
    
    def calculate_logits(self, lstm_outputs, batch_size, sequence_lengths):
        lstm_outputs_fw, lstm_outputs_bw = lstm_outputs[0], lstm_outputs[1]
        combined_outputs = tf.concat([lstm_outputs_fw, lstm_outputs_bw], -1)
        gather_indices = self.get_gather_indices(batch_size, sequence_lengths)
        final_outputs = tf.gather_nd(combined_outputs, gather_indices)
        logits = tf.keras.layers.Dense(1) (final_outputs)
        return logits

    def calculate_loss(self, lstm_outputs, batch_size, sequence_lengths, labels):
        logits = self.calculate_logits(lstm_outputs, batch_size, sequence_lengths)
        float_labels = tf.cast(labels, tf.float32)
        batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = float_labels, logits=logits)
        overall_loss = tf.reduce_sum(batch_loss)
        return overall_loss
    
    def logits_to_predictions(self, logits):
        probs = tf.math.sigmoid(logits)
        preds = tf.math.round(probs)
        return preds