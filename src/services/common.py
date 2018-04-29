import json
import pickle
import os.path

import numpy as np
import csv

from itertools import chain
import matplotlib.pyplot as plt

import constants.main_constants as const


def load_object(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def save_object(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_data(data_dir, split='train', debug=False):
    data_path = data_dir + split + '_tok.jsonl'
    table_path = data_dir + split + '_tok.tables.jsonl'
    db_path = data_dir + split + '.db'
    query_list = []
    sql_list = []
    table_data = {}
    with open(data_path) as f:
        for idx, line in enumerate(f):
            if debug and idx > const.DEBUG_DATA_SIZE:
                break
            data = json.loads(line.strip())
            query_list.append(data['question_tok'])
            sql_list.append(data['sql'])
    with open(table_path) as f:
        for _, line in enumerate(f):
            t_data = json.loads(line.strip())
            table_data[t_data['id']] = t_data
    return query_list, sql_list, table_data, db_path


def load_where_data(data_dir, split='train', debug=False):
    col_path = data_dir + split + '_tok_where_col.csv'
    op_path = data_dir + split + '_tok_where_op.csv'
    with open(col_path, 'r') as f:
        data = csv.reader(f)
        col_list = np.array([np.array(d[0].split(' '), dtype=int) for d in list(data)])
    with open(op_path, 'r') as f:
        data = csv.reader(f)
        op_list = np.array([np.array(d[0].split(' '), dtype=int) for d in list(data)])

    if debug:
        col_list = col_list[:const.DEBUG_DATA_SIZE]
        op_list = op_list[:const.DEBUG_DATA_SIZE]
    return col_list, op_list


def make_token_to_index(data, embedding, use_extra_tokens=True, load_if_exists=False, debug=False):
    if load_if_exists and os.path.exists(const.TOKEN_TO_IDX_SAVE) and os.path.exists(const.TOKEN_WEIGHTS_SAVE):
        token_to_index = load_object(const.TOKEN_TO_IDX_SAVE)
        token_weights = load_object(const.TOKEN_WEIGHTS_SAVE)
    else:
        idx = 0
        token_to_index = dict()
        token_weights = []
        if use_extra_tokens:
            token_to_index[const.UNK_TOKEN] = const.UNK_IDX
            token_weights.append(embedding.vector(const.UNK_TOKEN))
            token_to_index[const.BEG_TOKEN] = const.BEG_IDX
            token_weights.append(embedding.vector(const.BEG_TOKEN))
            token_to_index[const.END_TOKEN] = const.END_IDX
            token_weights.append(embedding.vector(const.END_TOKEN))
            idx += 3

        unique_tokens = set(chain.from_iterable(data))
        for token in unique_tokens:
            token_to_index[token] = idx
            token_weights.append(embedding.vector(token))
            idx += 1

        token_weights = np.array(token_weights)
        if not debug:
            save_object(token_to_index, const.TOKEN_TO_IDX_SAVE)
            save_object(token_weights, const.TOKEN_WEIGHTS_SAVE)
    return token_to_index, token_weights


def where_accuracy_score(true_col, true_op, predicted_col, predicted_op):
    batch_size = true_col.shape[0]
    correct = 0
    for idx in range(batch_size):
        if np.all(true_col[idx, ] == predicted_col[idx, ]) and np.all(true_op[idx, ] == predicted_op[idx, ]):
            correct += 1
    return correct/batch_size


def accuracy_plot(train_accuracy, val_accuracy, epochs, save_file):
    x = range(epochs)
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.plot(x, train_accuracy, label='Train Accuracy')
    ax.plot(x, val_accuracy, label='Validation Accuracy')
    ax.legend()
    plt.savefig('../plots/'+save_file)
