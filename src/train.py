from argparse import ArgumentParser

from models.data import DataModel
from models.model import NLQModel
from services.common import load_data, make_token_to_index
from services.logger import Logger
from models.glove import Glove

import constants.main_constants as const

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch Size')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning Rate')
parser.add_argument('--decay', type=float, default=0.95,
                    help='Decay for Learning Rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of Epochs')
parser.add_argument('--save', default='../save',
                    help='Model save directory.')
parser.add_argument('--save_every', type=int, default=500,
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
logger.end_timer()

glove = Glove(file_name=const.GLOVE, load_if_exists=(True and not args.hard_reload))
args.emb_size = glove.length
args.batch_size = const.BATCH_SIZE if not args.debug else const.DEBUG_BATCH_SIZE
args.load_if_exists = not args.hard_reload and not args.debug

logger.start_timer('Making token dictionary..')
token_to_index, token_weights = make_token_to_index(data=train_query_list, embedding=glove, use_extra_tokens=True, load_if_exists=(True and args.load_if_exists))
logger.end_timer()

data_model = DataModel(query_list=train_query_list, sql_list=train_sql_list, token_to_index=token_to_index)
validation_model = DataModel(query_list=dev_query_list, sql_list=dev_sql_list, token_to_index=token_to_index)

nlq_model = NLQModel(args, token_weights=token_weights)

nlq_model.start_train(data_model, validation_model)