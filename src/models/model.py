import os

import tensorflow as tf

from sklearn.metrics import accuracy_score

from models.predictors import AggregatePredictor


class NLQModel:
    def __init__(self, args, token_weights):
        self.args = args
        self.aggregate_predictor = AggregatePredictor(args, token_weights)

    def start_train(self, data_model, validation_model):
        args = self.args
        init = tf.global_variables_initializer()
        num_batches = data_model.total_batches
        tf_saver = tf.train.Saver(tf.global_variables())
        aggregate_predictor = self.aggregate_predictor
        with tf.Session() as sess:
            sess.run(init)
            for e in range(args.epochs):
                sess.run(tf.assign(aggregate_predictor.lr, args.lr * (args.decay ** e)))
                train_accuracy = 0
                for b in range(num_batches):
                    state = sess.run(aggregate_predictor.initial_state)
                    query, true_aggregate = data_model.next_batch()
                    feed = {
                        aggregate_predictor.input_data: query,
                        aggregate_predictor.targets: true_aggregate
                    }
                    for i, (c, h) in enumerate(aggregate_predictor.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    _, train_loss, state, predicted_output = sess.run([aggregate_predictor.optimizer, aggregate_predictor.cost, aggregate_predictor.final_state, aggregate_predictor.predicted_output], feed)
                    accuracy = 100 * accuracy_score(true_aggregate, predicted_output)
                    train_accuracy += accuracy
                    print("{}/{} : Epoch {} - Loss = {:.3f}, Accuracy = {:.2f}".format(e * num_batches + b, args.epochs * num_batches, e, train_loss, accuracy))
                    if (e * num_batches + b) % args.save_every == 0 or ( e == args.epochs - 1 and b == data_model.num_batches() - 1):
                        checkpoint_path = os.path.join(args.save, 'model.ckpt')
                        tf_saver.save(sess, checkpoint_path, global_step=e * num_batches + b)
                        print("Model saved to {}".format(checkpoint_path))

                print('Calculating Validation Accuracy..')
                validation_accuracy = 0
                for b in range(validation_model.total_batches):
                    state = sess.run(aggregate_predictor.initial_state)
                    query, true_aggregate = data_model.next_batch()
                    feed = {
                        aggregate_predictor.input_data: query,
                        aggregate_predictor.targets: true_aggregate
                    }
                    for i, (c, h) in enumerate(aggregate_predictor.initial_state):
                        feed[c] = state[i].c
                        feed[h] = state[i].h
                    _, train_loss, state, predicted_output = sess.run(
                        [aggregate_predictor.optimizer, aggregate_predictor.cost, aggregate_predictor.final_state,
                         aggregate_predictor.predicted_output], feed)
                    validation_accuracy += 100 * accuracy_score(true_aggregate, predicted_output)
                print('Train Accuracy = %.2f, Validation Accuracy = %.2f' % (train_accuracy/data_model.total_batches, validation_accuracy/validation_model.total_batches))