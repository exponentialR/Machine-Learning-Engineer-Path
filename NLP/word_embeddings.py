import tensorflow as tf
from tensorflow.python.ops.nn_impl import normalize

from NLP.utils.util_func import get_initializer

class EmbeddingModel(object):

    #Model Initialization 
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = self.vocab_size)

    
    #Conversion of list of text strings into word sequences 

    def tokenize_text_corpus(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return 
    

    def get_target_and_context(self, sequence, target_index, window_size):
        target_word = sequence[target_index]
        half_window_size = window_size //2
        left_incl = max(0, target_index - half_window_size)
        right_excl = min(len(sequence), target_index + half_window_size + 1)
        return target_word, left_incl, right_excl


    def create_target_context_pairs(self, texts, window_size):
        pairs = []
        sequences = self.tokenize_text_corpus(texts)
        for sequence in sequences:
            for i in range(len(sequence)):
                target_word, left_incl, right_excl = self.get_target_and_context(sequence, i, window_size)
                for j in range(left_incl, right_excl):
                    if j != i:
                        pairs.append((target_word, sequence[j]))
        return pairs


    def forward(self, target_ids):
        initializer = get_initializer(self.embedding_dim, self.vocab_size)
        self.embedding_matrix = tf.compat.v1.get_variable('embedding_matrix', initializer=initializer)
        embeddings = tf.compat.v1.nn.embedding_lookup(self.embedding_matrix, target_ids)
        return embeddings
    
    def get_bias_weights(self):
        weights_initializer = tf.zeros([self.vocab_size, self.embedding_dim])
        bias_initializer = tf.zeros([self.vocab_size])
        weights = tf.compat.v1.get_variable('weights', initializer=weights_initializer)
        bias = tf.compat.v1.get_variable('bias', initializer=bias_initializer)
        return weights, bias


    def calculate_loss(self, embeddings, context_ids, num_negative_samples):
        weights, bias = self.get_bias_weights()
        nce_losses = tf.nn.nce_loss(weights, bias, context_ids, embeddings, num_negative_samples, self.vocab_size)
        overall_loss = tf.math.reduce_mean(nce_losses)
        return overall_loss
    
    def compute_cos_sims(self, word, training_texts):
        self.tokenizer.fit_on_texts(training_texts)
        word_id = self.tokenizer.word_index[word]
        word_embedding = self.forward([word_id])
        normalized_embedding = tf.math.l2_normalize(word_embedding)
        normalized_matrix = tf.math.l2_normalize(self.embedding_matrix, axis = 1)
        cos_sims = tf.linalg.matmul(normalize, normalized_matrix, transpose_b = True)
        return cos_sims

    def k_nearest_neighbors(self, word, k, training_texts):
        cos_sims = self.compute_cos_sims(word, training_texts)
        squeezed_cos_sims = tf.squeeze(cos_sims)
        top_k_output = tf.math.top_k(squeezed_cos_sims, k)
        return top_k_output
