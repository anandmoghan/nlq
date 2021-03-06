import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits

import constants.main_constants as const


class AggregatePredictor:
    def __init__(self, args, token_weights):
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, const.MAX_LEN])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, 1])

        total_unique_tokens = len(token_weights)

        with tf.variable_scope("aggregate"):
            embedding = tf.get_variable("embedding", [total_unique_tokens, args.emb_size], initializer=tf.constant_initializer(token_weights))

            cells = []
            for _ in range(const.SEL_RNN_LAYERS):
                cells.append(rnn.BasicLSTMCell(const.SEL_RNN_SIZE))
            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_input = tf.split(input_embeddings, const.MAX_LEN, 1)
            rnn_input = [tf.squeeze(ip, [1]) for ip in rnn_input]

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            outputs, self.final_state = legacy_seq2seq.rnn_decoder(rnn_input, self.initial_state, cell)
            output = outputs[-1]
            hidden_state = self.final_state[const.AGG_RNN_LAYERS - 1].h
            out_logits = tf.contrib.layers.fully_connected(inputs=output, num_outputs=const.AGG_COUNT, activation_fn=None)
            ca_logits = tf.contrib.layers.fully_connected(inputs=hidden_state, num_outputs=const.AGG_COUNT, activation_fn=None)
            logits = out_logits + ca_logits
            self.probs = tf.nn.softmax(logits)
            self.predicted_output = tf.reshape(tf.argmax(self.probs, 1), [args.batch_size, 1])

            self.lr = tf.Variable(0.0, trainable=False)
            loss = sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.targets, [-1]))
            self.cost = tf.reduce_mean(loss)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)


class SelectPredictor:
    def __init__(self, args, token_weights):
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, const.MAX_LEN])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, 1])

        total_unique_tokens = len(token_weights)

        with tf.variable_scope("select"):
            embedding = tf.get_variable("embedding", [total_unique_tokens, args.emb_size], initializer=tf.constant_initializer(token_weights))

            cells = []
            for _ in range(const.SEL_RNN_LAYERS):
                cells.append(rnn.BasicLSTMCell(const.SEL_RNN_SIZE))
            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_input = tf.split(input_embeddings, const.MAX_LEN, 1)
            rnn_input = [tf.squeeze(ip, [1]) for ip in rnn_input]

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            outputs, self.final_state = legacy_seq2seq.rnn_decoder(rnn_input, self.initial_state, cell)
            output = outputs[-1]
            hidden_state = self.final_state[const.SEL_RNN_LAYERS - 1].h

            out_logits = tf.contrib.layers.fully_connected(inputs=output, num_outputs=const.SEL_COUNT, activation_fn=None)
            ca_logits = tf.contrib.layers.fully_connected(inputs=hidden_state, num_outputs=const.SEL_COUNT, activation_fn=None)
            logits = out_logits + ca_logits
            self.probs = tf.nn.softmax(logits)
            self.predicted_output = tf.reshape(tf.argmax(self.probs, 1), [args.batch_size, 1])

            self.lr = tf.Variable(0.0, trainable=False)
            loss = sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.reshape(self.targets, [-1]))
            self.cost = tf.reduce_mean(loss)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)


class WherePredictor:
    def __init__(self, args, token_weights):
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, const.MAX_LEN])
        self.col_targets = tf.placeholder(tf.int32, [args.batch_size, const.MAX_LEN])
        self.op_targets = tf.placeholder(tf.int32, [args.batch_size, const.MAX_LEN])

        total_unique_tokens = len(token_weights)

        with tf.variable_scope("where"):
            embedding = tf.get_variable("embedding", [total_unique_tokens, args.emb_size], initializer=tf.constant_initializer(token_weights))

            cells = []
            for _ in range(const.WHERE_RNN_LAYERS):
                cells.append(rnn.BasicLSTMCell(const.WHERE_RNN_SIZE))
            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_input = tf.split(input_embeddings, const.MAX_LEN, 1)
            rnn_input = [tf.squeeze(ip, [1]) for ip in rnn_input]

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            outputs, self.final_state = legacy_seq2seq.rnn_decoder(rnn_input, self.initial_state, cell)
            output = tf.reshape(tf.concat(outputs, 1), [-1, const.WHERE_RNN_SIZE])

            col_logits = tf.contrib.layers.fully_connected(inputs=output, num_outputs=const.WHERE_COL_COUNT, activation_fn=None)
            self.col_probs = tf.nn.softmax(col_logits)
            self.col_predicted_output = tf.reshape(tf.argmax(self.col_probs, 1), [args.batch_size, const.MAX_LEN])

            op_logits = tf.contrib.layers.fully_connected(inputs=output, num_outputs=const.WHERE_COL_COUNT, activation_fn=None)
            self.op_probs = tf.nn.softmax(op_logits)
            self.op_predicted_output = tf.reshape(tf.argmax(self.op_probs, 1), [args.batch_size, const.MAX_LEN])

            self.lr = tf.Variable(0.0, trainable=False)
            col_loss = sparse_softmax_cross_entropy_with_logits(logits=col_logits, labels=tf.reshape(self.col_targets, [-1]))
            op_loss = sparse_softmax_cross_entropy_with_logits(logits=op_logits, labels=tf.reshape(self.op_targets, [-1]))
            self.cost = tf.reduce_mean(col_loss + op_loss)
            self.cost = tf.reduce_mean(col_loss)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
