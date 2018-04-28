import numpy as np

import constants.main_constants as const


class DataModel:
    def __init__(self, query_list, sql_list, token_to_index, batch_size=const.BATCH_SIZE, shuffle=True):
        self.data_size = len(query_list)
        self.encoded_query_matrix = np.zeros([self.data_size, const.MAX_LEN])
        self.aggregate_array = np.zeros([self.data_size, 1])
        self.select_array = np.zeros([self.data_size, 1])
        for idx, (query, sql) in enumerate(zip(query_list, sql_list)):
            encoded_query = [const.BEG_IDX] + list(map(lambda x: token_to_index.get(x, const.UNK_IDX), query)) + [const.END_IDX]
            if sql['sel'] < 22:
                self.encoded_query_matrix[idx, const.MAX_LEN - len(encoded_query):] = encoded_query
                self.aggregate_array[idx] = sql['agg']
                self.select_array[idx] = sql['sel']
        self.current_batch = 1
        self.batch_size = batch_size
        self.total_batches = int(self.data_size/batch_size)
        self.shuffle = shuffle

    def reset_pointer(self):
        self.current_batch = 1
        if self.shuffle:
            shuffle_idx = np.random.permutation(self.data_size)
            self.encoded_query_matrix = self.encoded_query_matrix[shuffle_idx, ]
            self.aggregate_array = self.aggregate_array[shuffle_idx, ]
            self.select_array = self.select_array[shuffle_idx,]

    def next_batch(self):
        batch_idx = np.arange((self.current_batch - 1) * self.batch_size, self.current_batch * self.batch_size)
        self.current_batch += 1
        if self.current_batch > self.total_batches:
            self.reset_pointer()
        return self.encoded_query_matrix[batch_idx, ], self.aggregate_array[batch_idx, ], self.select_array[batch_idx, ]

