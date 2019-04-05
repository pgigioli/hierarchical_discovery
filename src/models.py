#!/usr/bin/env python

import tensorflow as tf

from . import layers
from . import encoder
                
class RNNClassifier:
    def __init__(self, n_classes, vocab_size, max_len, learning_rate=0.001, embedding_dim=100, hidden_size=128, use_attn=False, n_layers=1,
                pretrained_embeddings=None, trainable_embeddings=True, rnn_cell=tf.contrib.rnn.GRUCell, class_weights=None):
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.use_attn = use_attn
        self.n_layers = n_layers
        self.trainable_embeddings = trainable_embeddings
        self.rnn_cell = rnn_cell
        self.class_weights = class_weights
        self._pretrained_embeddings = pretrained_embeddings
        
        self._build_graph()
        
    def _placeholders(self):
        self.inputs = tf.placeholder(tf.int32, [None, self.max_len], name='inputs')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
        self.targets = tf.placeholder(tf.int32, [None], name='targets')
        
    def _cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.rnn_cell(num_units=self.hidden_size), 
                                             input_keep_prob=self.dropout_keep_prob,
                                             output_keep_prob=self.dropout_keep_prob)
    def _embeddings_layer(self):
        self.embeddings = layers.embeddings_layer(self.vocab_size, self.embedding_dim, 
                                                  trainable=self.trainable_embeddings,
                                                  pretrained_embeddings=self._pretrained_embeddings,
                                                  name='embeddings')
        self.inputs_embd = tf.nn.embedding_lookup(self.embeddings, self.inputs)
        
    def _encoder(self):
        if self.n_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([self._cell() for _ in range(self.n_layers)])
        else:
            cell = self._cell()

        self.rnn_outputs, self.rnn_state = encoder.rnn_encoder(self.inputs_embd, cell, input_lens=self.input_lens)
        
        if self.use_attn:
            self.rnn_state, self.attn_weights = layers.self_attention(self.rnn_outputs, seq_lens=self.input_lens, get_variable=False)
        
    def _output_layer(self):
        self.logits = layers.dense(self.rnn_state, self.n_classes)
        
    def _train_ops(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.predictions = tf.to_int32(tf.argmax(self.logits, axis=1))
        self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.predictions, self.targets)))
        
        if self.class_weights is not None:
            class_weights = tf.gather(self.class_weights, self.targets)
        else:
            class_weights = 1.0
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.targets, self.logits, weights=class_weights))

        t_vars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), 5.0)
        self.gradient_norm = tf.global_norm(gradients)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, global_step=self.global_step)
        
    def _build_graph(self):
        with tf.variable_scope('placeholders'):
            self._placeholders()
        
        with tf.variable_scope('embeddings_layer'):
            self._embeddings_layer()
        
        with tf.variable_scope('encoder'):
            self._encoder()
            
        with tf.variable_scope('output_layer'):
            self._output_layer()
            
        with tf.variable_scope('train_ops'):
            self._train_ops()
    
class DiscreteHierarchicalClassifier(RNNClassifier):
    def __init__(self, n_classes, vocab_size, max_len, learning_rate=0.001, embedding_dim=100, hidden_size=128, straight_through=False, use_attn=False, n_layers=1,
                 hierarchies=None, pretrained_embeddings=None, trainable_embeddings=True, rnn_cell=tf.contrib.rnn.GRUCell, class_weights=None):
        self.straight_through = straight_through
        self.hierarchies = hierarchies
        
        if self.hierarchies == None: self.hierarchies = []
        self.n_hierarchies = len(self.hierarchies)
        
        super(DiscreteHierarchicalClassifier, self).__init__(
            n_classes, vocab_size, max_len, learning_rate=learning_rate, embedding_dim=embedding_dim, hidden_size=hidden_size, use_attn=use_attn, n_layers=n_layers,
            pretrained_embeddings=pretrained_embeddings, trainable_embeddings=trainable_embeddings, rnn_cell=rnn_cell, class_weights=class_weights
        )
        
    def _placeholders(self):
        self.inputs = tf.placeholder(tf.int32, [None, self.max_len], name='inputs')
        self.input_lens = tf.placeholder(tf.int32, [None], name='input_lens')
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
        self.gumbel_temperature = tf.placeholder(tf.float32, shape=(), name='gumbel_temperature')
        self.discrete_loss_weight = tf.placeholder(tf.float32, shape=(), name='discrete_loss_weight')
        self.continuous_loss_weight = tf.placeholder(tf.float32, shape=(), name='continuous_loss_weight')
        self.gumbel_loss_weight = tf.placeholder(tf.float32, shape=(), name='gumbel_loss_weight')
        self.orthogonality_loss_weight = tf.placeholder(tf.float32, shape=(), name='orthogonality_loss_weight')
        self.mse_loss_weight = tf.placeholder(tf.float32, shape=(), name='mse_loss_weight')
        self.targets = tf.placeholder(tf.int32, [None], name='targets')
        
    def _hierarchical_layer(self):
        self.clean_hierarchical_state, self.noisy_hierarchical_state = self.rnn_state, self.rnn_state
        self.clean_states, self.noisy_states, self.gumbel_softmaxes, self.orthogonalities, self.gumbel_onehots = [], [], [], [], []
        for i in range(self.n_hierarchies):
            # clean
            clean_dense = layers.dense(self.clean_hierarchical_state, self.hierarchies[i], bias=True, activation=None, name='dense_{}'.format(i), get_variable=True)
            clean_softmax = layers.gumbel_softmax(clean_dense, temperature=self.gumbel_temperature, straight_through=True)
            clean_state, orthogonality = layers.orth_dense(clean_softmax, self.hidden_size, bias=True, activation=tf.tanh, name='gumbel_dense_{}'.format(i), get_variable=True)
            self.clean_hierarchical_state = tf.nn.dropout(self.clean_hierarchical_state, self.dropout_keep_prob) + clean_state
            self.clean_states.append(clean_softmax)
            
            # noisy
            noisy_dense = layers.dense(self.noisy_hierarchical_state, self.hierarchies[i], bias=True, activation=None, name='dense_{}'.format(i), get_variable=True)
            noisy_softmax = layers.gumbel_softmax(noisy_dense, temperature=self.gumbel_temperature, straight_through=False)
            noisy_state, _ = layers.orth_dense(noisy_softmax, self.hidden_size, bias=True, activation=tf.tanh, name='gumbel_dense_{}'.format(i), get_variable=True)
            self.noisy_hierarchical_state = tf.nn.dropout(self.noisy_hierarchical_state, self.dropout_keep_prob) + noisy_state
            self.noisy_states.append(noisy_softmax)

            self.gumbel_softmaxes.append(noisy_softmax)
            self.gumbel_onehots.append(clean_softmax)
            self.orthogonalities.append(orthogonality)
         
    def _decoder(self):
        self.denoised_states = []
        denoised_state = self.continuous_logits
        for i, noisy_state in enumerate(self.noisy_states[::-1]):
            denoised_state = tf.tanh(
                layers.dense(noisy_state, self.hidden_size, bias=False, activation=None, name='denoise1_{}'.format(i), get_variable=True)
                + layers.dense(denoised_state, self.hidden_size, bias=True, activation=None, name='denoise2_{}'.format(i), get_variable=True)
            )
            self.denoised_states.append(denoised_state)
        self.recon_state = layers.dense(denoised_state, self.hidden_size, bias=True, activation=tf.tanh, name='recon', get_variable=True)
        self.denoised_states.append(self.recon_state)
        self.clean_states = [self.rnn_state] + self.clean_states
        self.denoised_states = self.denoised_states[::-1]

    def _output_layer(self):
        if self.straight_through:
            hierarchical_state = self.clean_hierarchical_state
        else:
            hierarchical_state = self.noisy_hierarchical_state
            
        self.discrete_logits = layers.dense(
            layers.dense(tf.concat(self.clean_states, axis=-1), self.hidden_size, bias=True, activation=tf.tanh, name='output_concat', get_variable=True), 
            self.n_classes, name='output_projection', bias=True, activation=None, get_variable=True
        )
        self.continuous_logits = layers.dense(hierarchical_state, self.n_classes, name='output_projection', bias=True, activation=None, get_variable=True)

    def _train_ops(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.continuous_predictions = tf.to_int32(tf.argmax(self.continuous_logits, axis=1))
        self.discrete_predictions = tf.to_int32(tf.argmax(self.discrete_logits, axis=1))
        self.continuous_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.continuous_predictions, self.targets)))
        self.discrete_accuracy = tf.reduce_mean(tf.to_float(tf.equal(self.discrete_predictions, self.targets)))
        
        # classification loss
        if self.class_weights is not None:
            class_weights = tf.gather(self.class_weights, self.targets)
        else:
            class_weights = 1.0        
        self.continuous_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.targets, self.continuous_logits, weights=class_weights))
        self.discrete_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(self.targets, self.discrete_logits, weights=class_weights))
                
        # gumbel loss
        self.gumbel_loss = 0.0
        for gumbel_softmax in self.gumbel_softmaxes:
            self.gumbel_loss += tf.reduce_mean(tf.reduce_sum(gumbel_softmax*(tf.log(gumbel_softmax+1e-20)-tf.log(1/self.n_classes)), axis=1))
        self.gumbel_loss /= self.n_hierarchies
        
        # orthogonality loss
        self.orthogonality_loss = tf.reduce_mean(self.orthogonalities)
        
        # mean squared error loss        
        self.mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.rnn_state, self.recon_state))

        # total loss
        self.loss = (
            self.continuous_loss_weight*self.continuous_loss 
            + self.discrete_loss_weight*self.discrete_loss
            + self.gumbel_loss_weight*self.gumbel_loss 
            + self.orthogonality_loss_weight*self.orthogonality_loss 
            + self.mse_loss_weight*self.mse_loss
        )

        t_vars = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, t_vars), 5.0)
        self.gradient_norm = tf.global_norm(gradients)
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = opt.minimize(self.loss, global_step=self.global_step)
        self.pretrain_optimizer = opt.minimize(self.loss, global_step=self.global_step, var_list=self.pretrain_vars)
        self.finetune_optimizer = opt.minimize(self.loss, global_step=self.global_step, var_list=self.finetune_vars)
        
    def _build_graph(self):
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            with tf.variable_scope('placeholders'):
                self._placeholders()

            with tf.variable_scope('embeddings_layer'):
                self._embeddings_layer()

            with tf.variable_scope('encoder'):
                self._encoder()

            self.pretrain_vars = tf.trainable_variables()

            with tf.variable_scope('hierarchical_layer'):
                self._hierarchical_layer()

            with tf.variable_scope('output_layer'):
                self._output_layer()
                
            with tf.variable_scope('decoder'):
                self._decoder()

            self.finetune_vars = [x for x in tf.trainable_variables() if x not in self.pretrain_vars]

            with tf.variable_scope('train_ops'):
                self._train_ops()
                
class DiscreteAttnHierarchicalClassifier(DiscreteHierarchicalClassifier):
    def __init__(self, n_classes, vocab_size, max_len, learning_rate=0.001, embedding_dim=100, hidden_size=128, straight_through=False, use_attn=False, n_layers=1,
                 hierarchies=None, pretrained_embeddings=None, trainable_embeddings=True, rnn_cell=tf.contrib.rnn.GRUCell, class_weights=None):
        
        super(DiscreteAttnHierarchicalClassifier, self).__init__(
            n_classes, vocab_size, max_len, learning_rate=learning_rate, embedding_dim=embedding_dim, hidden_size=hidden_size, straight_through=straight_through, use_attn=use_attn, 
            n_layers=n_layers, hierarchies=hierarchies, pretrained_embeddings=pretrained_embeddings, trainable_embeddings=trainable_embeddings, rnn_cell=rnn_cell, class_weights=class_weights
        )

    def _hierarchical_layer(self):
        self.clean_hierarchical_state, self.noisy_hierarchical_state = self.rnn_state, self.rnn_state
        self.clean_states, self.noisy_states, self.gumbel_softmaxes, self.orthogonalities, self.gumbel_onehots, self.attn_weights = [], [], [], [], [], []
        for i in range(self.n_hierarchies):
            # clean
            clean_dense = layers.dense(self.clean_hierarchical_state, self.hierarchies[i], bias=True, activation=None, name='dense_{}'.format(i), get_variable=True)
            clean_softmax = layers.gumbel_softmax(clean_dense, temperature=self.gumbel_temperature, straight_through=True)
            clean_state, orthogonality = layers.orth_dense(clean_softmax, self.hidden_size, bias=True, activation=tf.tanh, name='gumbel_dense_{}'.format(i), get_variable=True)
            clean_attn_state, attn_weight = layers.attention(self.rnn_outputs, clean_softmax, seq_lens=self.input_lens, name="attn_{}".format(i), get_variable=True)
            self.clean_hierarchical_state = tf.nn.dropout(self.clean_hierarchical_state, self.dropout_keep_prob) + clean_attn_state
            self.clean_states.append(clean_softmax)
            
            # noisy
            noisy_dense = layers.dense(self.noisy_hierarchical_state, self.hierarchies[i], bias=True, activation=None, name='dense_{}'.format(i), get_variable=True)
            noisy_softmax = layers.gumbel_softmax(noisy_dense, temperature=self.gumbel_temperature, straight_through=False)
            noisy_state, _ = layers.orth_dense(noisy_softmax, self.hidden_size, bias=True, activation=tf.tanh, name='gumbel_dense_{}'.format(i), get_variable=True)
            noisy_attn_state, _ = layers.attention(self.rnn_outputs, noisy_softmax, seq_lens=self.input_lens, name="attn_{}".format(i), get_variable=True)
            self.noisy_hierarchical_state = tf.nn.dropout(self.noisy_hierarchical_state, self.dropout_keep_prob) + noisy_attn_state
            self.noisy_states.append(noisy_softmax)

            self.gumbel_softmaxes.append(noisy_softmax)
            self.gumbel_onehots.append(clean_softmax)
            self.orthogonalities.append(orthogonality)
            self.attn_weights.append(attn_weight)