from argparse import ArgumentParser

from models.data import DataModel
from models.model import NLQModel
from services.common import load_data, make_token_to_index, load_where_data, save_object
from services.logger import Logger
from models.glove import Glove

import constants.main_constants as const

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256,
                    help='Mini Batch Size')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of Epochs')
parser.add_argument('--save', default='../save',
                    help='Model save directory.')
parser.add_argument('--save_every', type=int, default=1000,
                    help='Model save frequency')
parser.add_argument('--debug', action='store_true',
                    help='Fast debugging mode.')
parser.add_argument('--hard_reload', action='store_true',
                    help='Pre-processing will be done from scratch.')
args = parser.parse_args()
logger = Logger()

# TODO: Make a components check function in services.common to check for data, glove and directories.

logger.start_timer('Loading data..')
train_query_list, train_sql_list, train_table_data, train_db = load_data(data_dir=const.DATA_DIR, split='train', debug=args.debug)
dev_query_list, dev_sql_list, dev_table_data, dev_db = load_data(data_dir=const.DATA_DIR, split='dev', debug=args.debug)
test_query_list, test_sql_list, test_table_data, test_db = load_data(data_dir=const.DATA_DIR, split='test', debug=args.debug)
train_where_col_list, train_where_op_list = load_where_data(data_dir=const.DATA_DIR, split='train', debug=args.debug)
dev_where_col_list, dev_where_op_list = load_where_data(data_dir=const.DATA_DIR, split='dev', debug=args.debug)
test_where_col_list, test_where_op_list = load_where_data(data_dir=const.DATA_DIR, split='test', debug=args.debug)
logger.end_timer()


glove = Glove(file_name=const.GLOVE, load_if_exists=(True and not args.hard_reload))
args.emb_size = glove.length
args.batch_size = args.batch_size if not args.debug else const.DEBUG_BATCH_SIZE
args.load_if_exists = not args.hard_reload and not args.debug

logger.start_timer('Making token dictionary..')
token_to_index, token_weights = make_token_to_index(data=train_query_list, embedding=glove, use_extra_tokens=True, load_if_exists=(True and args.load_if_exists))
logger.end_timer()

logger.start_timer('Making Data Model..')
data_model = DataModel(query_list=train_query_list, sql_list=train_sql_list, where_col_list=train_where_col_list, where_op_list=train_where_op_list, token_to_index=token_to_index, batch_size=args.batch_size)
validation_model = DataModel(query_list=dev_query_list, sql_list=dev_sql_list, where_col_list=dev_where_col_list, where_op_list=dev_where_op_list, token_to_index=token_to_index, batch_size=args.batch_size)
logger.end_timer()

nlq_model = NLQModel(args, token_weights=token_weights, train_choice=const.TRAIN_SEL)

aggregate_accuracy, select_accuracy, where_accuracy = nlq_model.train(data_model, validation_model)
save_object(aggregate_accuracy, file_name='../plots/agg_acc.pkl')
save_object(select_accuracy, file_name='../plots/sel_acc.pkl')
save_object(where_accuracy, file_name='../plots/where_acc.pkl')
