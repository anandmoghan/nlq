import os

import tensorflow as tf
import numpy as np

from sklearn.metrics import accuracy_score

from models.predictors import AggregatePredictor, SelectPredictor, WherePredictor
from services.common import where_accuracy_score
from services.logger import Logger

import constants.main_constants as const


GPU_OPTIONS = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


class NLQModel:

    def __init__(self, args, token_weights, train_choice=const.TRAIN_ALL):
        self.args = args
        self.aggregate_predictor = AggregatePredictor(args, token_weights)
        self.select_predictor = SelectPredictor(args, token_weights)
        self.where_predictor = WherePredictor(args, token_weights)
        self.train_agg, self.train_sel, self.train_where = train_choice

    def train(self, data_model, validation_model):
        args = self.args
        init = tf.global_variables_initializer()
        num_batches = data_model.total_batches
        tf_saver = tf.train.Saver(tf.global_variables())
        aggregate_predictor = self.aggregate_predictor
        select_predictor = self.select_predictor
        where_predictor = self.where_predictor
        logger = Logger()
        train_aggregate_accuracy = np.zeros(args.epochs)
        train_select_accuracy = np.zeros(args.epochs)
        train_where_accuracy = np.zeros(args.epochs)
        validation_aggregate_accuracy = np.zeros(args.epochs)
        validation_select_accuracy = np.zeros(args.epochs)
        validation_where_accuracy = np.zeros(args.epochs)
        with tf.Session(config=tf.ConfigProto(gpu_options=GPU_OPTIONS)) as sess:
            sess.run(init)
            for e in range(args.epochs):
                logger.start_timer('Epoch: {:d} Training..'.format(e + 1))
                sess.run(tf.assign(aggregate_predictor.lr, const.AGG_LR * (const.AGG_DECAY ** e)))
                sess.run(tf.assign(select_predictor.lr, const.SEL_LR * (const.SEL_DECAY ** e)))
                sess.run(tf.assign(where_predictor.lr, const.WHERE_LR * (const.WHERE_DECAY ** e)))

                for b in range(num_batches):
                    print("{}/{} : Epoch {} - Learning Rate: {}".format(e * num_batches + b + 1, args.epochs * num_batches, e + 1, const.AGG_LR * (const.AGG_DECAY ** e)))
                    query, true_aggregate, true_select, true_where_col, true_where_op = data_model.next_batch()

                    if self.train_agg:
                        aggregate_state = sess.run(aggregate_predictor.initial_state)
                        aggregate_feed = {
                            aggregate_predictor.input_data: query,
                            aggregate_predictor.targets: true_aggregate
                        }

                        for i, (c, h) in enumerate(aggregate_predictor.initial_state):
                            aggregate_feed[c] = aggregate_state[i].c
                            aggregate_feed[h] = aggregate_state[i].h

                        _, aggregate_train_loss, _, predicted_output = sess.run([aggregate_predictor.optimizer, aggregate_predictor.cost, aggregate_predictor.final_state, aggregate_predictor.predicted_output], feed_dict=aggregate_feed)
                        aggregate_accuracy = 100 * accuracy_score(true_aggregate, predicted_output)
                        train_aggregate_accuracy[e] += aggregate_accuracy

                        print("Aggregate: Loss = {:.3f}, Accuracy = {:.2f}".format(aggregate_train_loss, aggregate_accuracy))

                    if self.train_sel:
                        select_state = sess.run(select_predictor.initial_state)
                        select_feed = {
                            select_predictor.input_data: query,
                            select_predictor.targets: true_select
                        }

                        for i, (c, h) in enumerate(select_predictor.initial_state):
                            select_feed[c] = select_state[i].c
                            select_feed[h] = select_state[i].h

                        _, select_train_loss, _, predicted_output = sess.run([select_predictor.optimizer, select_predictor.cost, select_predictor.final_state, select_predictor.predicted_output], feed_dict=select_feed)
                        select_accuracy = 100 * accuracy_score(true_select, predicted_output)
                        train_select_accuracy[e] += select_accuracy

                        print("Select: Loss = {:.3f}, Accuracy = {:.2f}".format(select_train_loss, select_accuracy))

                    if self.train_where:
                        where_state = sess.run(where_predictor.initial_state)
                        where_feed = {
                            where_predictor.input_data: query,
                            where_predictor.col_targets: true_where_col,
                            where_predictor.op_targets: true_where_op
                        }

                        for i, (c, h) in enumerate(where_predictor.initial_state):
                            where_feed[c] = where_state[i].c
                            where_feed[h] = where_state[i].h

                        _, where_train_loss, _, col_predicted_output, op_predicted_output = sess.run([where_predictor.optimizer, where_predictor.cost, where_predictor.final_state, where_predictor.col_predicted_output, where_predictor.op_predicted_output], feed_dict=where_feed)
                        where_accuracy = 100 * where_accuracy_score(true_where_op, true_where_op, col_predicted_output, op_predicted_output)
                        train_where_accuracy[e] += where_accuracy

                        print("Where: Loss = {:.3f}, Accuracy = {:.2f}".format(where_train_loss, where_accuracy))

                    if (e * num_batches + b + 1) % args.save_every == 0 or (e == args.epochs - 1 and b == data_model.total_batches - 1):
                        checkpoint_path = os.path.join(args.save, 'model.ckpt')
                        tf_saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                        print("Model saved to {}".format(checkpoint_path))

                logger.end_timer('Epoch: {:d}'.format(e + 1))
                logger.start_timer('Calculating Validation Accuracy..')

                for b in range(validation_model.total_batches):
                    query, true_aggregate, true_select, true_where_col, true_where_op = validation_model.next_batch()

                    if self.train_agg:
                        aggregate_state = sess.run(aggregate_predictor.initial_state)
                        aggregate_feed = {
                            aggregate_predictor.input_data: query,
                            aggregate_predictor.targets: true_aggregate
                        }

                        for i, (c, h) in enumerate(aggregate_predictor.initial_state):
                            aggregate_feed[c] = aggregate_state[i].c
                            aggregate_feed[h] = aggregate_state[i].h

                        _, _, predicted_output = sess.run([aggregate_predictor.cost, aggregate_predictor.final_state, aggregate_predictor.predicted_output], feed_dict=aggregate_feed)
                        validation_aggregate_accuracy[e] += 100 * accuracy_score(true_aggregate, predicted_output)

                    if self.train_sel:
                        select_state = sess.run(select_predictor.initial_state)
                        select_feed = {
                            select_predictor.input_data: query,
                            select_predictor.targets: true_select
                        }

                        for i, (c, h) in enumerate(select_predictor.initial_state):
                            select_feed[c] = select_state[i].c
                            select_feed[h] = select_state[i].h

                        _, _, predicted_output = sess.run([select_predictor.cost, select_predictor.final_state, select_predictor.predicted_output], feed_dict=select_feed)
                        validation_select_accuracy[e] += 100 * accuracy_score(true_select, predicted_output)

                    if self.train_where:
                        where_state = sess.run(where_predictor.initial_state)
                        where_feed = {
                            where_predictor.input_data: query,
                            where_predictor.col_targets: true_where_col,
                            where_predictor.op_targets: true_where_op
                        }

                        for i, (c, h) in enumerate(where_predictor.initial_state):
                            where_feed[c] = where_state[i].c
                            where_feed[h] = where_state[i].h

                        _, where_train_loss, _, col_predicted_output, op_predicted_output = sess.run([where_predictor.optimizer, where_predictor.cost, where_predictor.final_state, where_predictor.col_predicted_output, where_predictor.op_predicted_output], feed_dict=where_feed)
                        where_accuracy = 100 * where_accuracy_score(true_where_op, true_where_op, col_predicted_output, op_predicted_output)
                        validation_where_accuracy[e] += where_accuracy

                logger.end_timer()

                if self.train_agg:
                    train_aggregate_accuracy[e] /= data_model.total_batches
                    validation_aggregate_accuracy[e] /= validation_model.total_batches
                    print('Aggregate: Train Accuracy: %.2f | Validation Accuracy: %.2f' % (train_aggregate_accuracy[e], validation_aggregate_accuracy[e]))

                if self.train_sel:
                    train_select_accuracy[e] /= data_model.total_batches
                    validation_select_accuracy[e] /= validation_model.total_batches
                    print('Select: Train Accuracy: %.2f | Validation Accuracy: %.2f' % (train_select_accuracy[e], validation_select_accuracy[e]))

                if self.train_where:
                    train_where_accuracy[e] /= data_model.total_batches
                    validation_where_accuracy[e] /= validation_model.total_batches
                    print('Where: Train Accuracy: %.2f | Validation Accuracy: %.2f' % (train_where_accuracy[e], validation_where_accuracy[e]))

        return (train_aggregate_accuracy, validation_aggregate_accuracy), (train_select_accuracy, validation_select_accuracy), (train_where_accuracy, validation_where_accuracy)

    def test(self, test_model):
        args = self.args
        init = tf.global_variables_initializer()
        tf_saver = tf.train.Saver(tf.global_variables())
        aggregate_predictor = self.aggregate_predictor
        select_predictor = self.select_predictor
        where_predictor = self.where_predictor
        logger = Logger()
        with tf.Session(config=tf.ConfigProto(gpu_options=GPU_OPTIONS)) as sess:
            sess.run(init)
            checkpoint = tf.train.get_checkpoint_state(args.save)
            if checkpoint and checkpoint.model_checkpoint_path:
                tf_saver.restore(sess, checkpoint.model_checkpoint_path)
            test_aggregate_accuracy = 0
            test_select_accuracy = 0
            test_where_accuracy = 0
            logger.start_timer('Calculating Test Accuracy')
            for b in range(test_model.total_batches):
                query, true_aggregate, true_select, true_where_col, true_where_op = test_model.next_batch()

                aggregate_state = sess.run(aggregate_predictor.initial_state)
                aggregate_feed = {
                    aggregate_predictor.input_data: query,
                    aggregate_predictor.targets: true_aggregate
                }

                for i, (c, h) in enumerate(aggregate_predictor.initial_state):
                    aggregate_feed[c] = aggregate_state[i].c
                    aggregate_feed[h] = aggregate_state[i].h

                _, _, predicted_output = sess.run([aggregate_predictor.cost, aggregate_predictor.final_state, aggregate_predictor.predicted_output], feed_dict=aggregate_feed)
                test_aggregate_accuracy += 100 * accuracy_score(true_aggregate, predicted_output)

                select_state = sess.run(select_predictor.initial_state)
                select_feed = {
                    select_predictor.input_data: query,
                    select_predictor.targets: true_select
                }

                for i, (c, h) in enumerate(select_predictor.initial_state):
                    select_feed[c] = select_state[i].c
                    select_feed[h] = select_state[i].h

                _, _, predicted_output = sess.run([select_predictor.cost, select_predictor.final_state, select_predictor.predicted_output], feed_dict=select_feed)
                test_select_accuracy += accuracy_score(true_select, predicted_output)

                where_state = sess.run(select_predictor.initial_state)
                where_feed = {
                    where_predictor.input_data: query,
                    where_predictor.col_targets: true_where_col,
                    where_predictor.op_targets: true_where_op
                }

                for i, (c, h) in enumerate(where_predictor.initial_state):
                    where_feed[c] = where_state[i].c
                    where_feed[h] = where_state[i].h

                _, _, col_predicted_output, op_predicted_output = sess.run([where_predictor.cost, where_predictor.final_state, where_predictor.col_predicted_output, where_predictor.op_predicted_output], feed_dict=where_feed)
                test_where_accuracy += 100 * where_accuracy_score(true_where_op, true_where_op, col_predicted_output, op_predicted_output)

            test_aggregate_accuracy /= test_model.total_batches
            test_select_accuracy /= test_model.total_batches
            test_where_accuracy /= test_model.total_batches
            logger.end_timer()
            print('Aggregate Test Accuracy = %.2f' % test_aggregate_accuracy)
            print('Select Test Accuracy = %.2f' % test_select_accuracy)
            print('Where Test Accuracy = %.2f' % test_where_accuracy)
