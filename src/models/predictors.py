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
            for _ in range(const.AGG_RNN_LAYERS):
                cells.append(rnn.BasicLSTMCell(const.AGG_RNN_SIZE))
            self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

            dense_layer_w = tf.get_variable("dense_layer_w", [const.AGG_RNN_SIZE, const.AGG_COUNT])
            dense_layer_b = tf.get_variable("dense_layer_b", [const.AGG_COUNT])

            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_input = tf.split(input_embeddings, const.MAX_LEN, 1)
            rnn_input = [tf.squeeze(ip, [1]) for ip in rnn_input]

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            outputs, self.final_state = legacy_seq2seq.rnn_decoder(rnn_input, self.initial_state, cell)
            output = outputs[-1]
            logits = tf.matmul(output, dense_layer_w) + dense_layer_b
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

            dense_layer_w = tf.get_variable("dense_layer_w", [const.SEL_RNN_SIZE, const.SEL_COUNT])
            dense_layer_b = tf.get_variable("dense_layer_b", [const.SEL_COUNT])

            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_input = tf.split(input_embeddings, const.MAX_LEN, 1)
            rnn_input = [tf.squeeze(ip, [1]) for ip in rnn_input]

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            outputs, self.final_state = legacy_seq2seq.rnn_decoder(rnn_input, self.initial_state, cell)
            output = outputs[-1]
            logits = tf.matmul(output, dense_layer_w) + dense_layer_b
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

            col_dense_layer_w = tf.get_variable("col_dense_layer_w", [const.WHERE_RNN_SIZE, const.WHERE_COL_COUNT])
            col_dense_layer_b = tf.get_variable("col_dense_layer_b", [const.WHERE_COL_COUNT])

            op_dense_layer_w = tf.get_variable("op_dense_layer_w", [const.WHERE_RNN_SIZE, const.WHERE_OP_COUNT])
            op_dense_layer_b = tf.get_variable("op_dense_layer_b", [const.WHERE_OP_COUNT])

            input_embeddings = tf.nn.embedding_lookup(embedding, self.input_data)
            rnn_input = tf.split(input_embeddings, const.MAX_LEN, 1)
            rnn_input = [tf.squeeze(ip, [1]) for ip in rnn_input]

            self.initial_state = cell.zero_state(args.batch_size, tf.float32)
            outputs, self.final_state = legacy_seq2seq.rnn_decoder(rnn_input, self.initial_state, cell)
            output = tf.reshape(tf.concat(outputs, 1), [-1, const.WHERE_RNN_SIZE])

            col_logits = tf.matmul(output, col_dense_layer_w) + col_dense_layer_b
            self.col_probs = tf.nn.softmax(col_logits)
            self.col_predicted_output = tf.reshape(tf.argmax(self.col_probs, 1), [args.batch_size, const.MAX_LEN])

            op_logits = tf.matmul(output, op_dense_layer_w) + op_dense_layer_b
            self.op_probs = tf.nn.softmax(op_logits)
            self.op_predicted_output = tf.reshape(tf.argmax(self.op_probs, 1), [args.batch_size, const.MAX_LEN])

            self.lr = tf.Variable(0.0, trainable=False)
            col_loss = sparse_softmax_cross_entropy_with_logits(logits=col_logits, labels=tf.reshape(self.col_targets, [-1]))
            op_loss = sparse_softmax_cross_entropy_with_logits(logits=op_logits, labels=tf.reshape(self.op_targets, [-1]))
            self.cost = tf.reduce_mean(col_loss + op_loss)
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
