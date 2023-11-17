import copy

import yaml
from utils.log_utils import cus_logger
import copy

def read_yaml(cfg_name):
    cfg_root = 'config/'
    path = cfg_root + cfg_name
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict


def print_dict(args):
    logger2 = cus_logger(args, __name__)
    dict_info = copy.deepcopy(args)

    def info_dict(dict_in):
        for key, value in dict_in.items():
            if isinstance(value, dict):
                info_dict(value)
            else:
                logger2.info(f'{key} : {value}')

    info_dict(dict_info)
