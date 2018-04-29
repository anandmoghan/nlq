from argparse import ArgumentParser

from models.data import DataModel
from models.glove import Glove
from models.model import NLQModel
from services.common import load_data, make_token_to_index, load_where_data

import constants.main_constants as const
from services.logger import Logger

parser = ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64,
                    help='Batch Size')
parser.add_argument('--save', default='../save',
                    help='Model save directory.')
args = parser.parse_args()
logger = Logger()

logger.start_timer('Loading data..')
test_query_list, test_sql_list, test_table_data, test_db = load_data(data_dir=const.DATA_DIR, split='test')
test_where_col_list, test_where_op_list = load_where_data(data_dir=const.DATA_DIR, split='test')
logger.end_timer()

glove = Glove(file_name=const.GLOVE, load_if_exists=True)
args.emb_size = glove.length
args.batch_size = const.BATCH_SIZE

logger.start_timer('Loading token dictionary..')
token_to_index, token_weights = make_token_to_index(data=None, embedding=None, load_if_exists=True)
logger.end_timer()

logger.start_timer('Making Data Model..')
test_model = DataModel(query_list=test_query_list, sql_list=test_sql_list, where_col_list=test_where_col_list, where_op_list=test_where_op_list, token_to_index=token_to_index)
logger.end_timer()

nlq_model = NLQModel(args, token_weights=token_weights)
nlq_model.test(test_model)