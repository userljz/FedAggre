import copy
import yaml
from Utils.log_utils import cus_logger

def read_yaml(cfg_name):
    cfg_root = 'config/'
    path = cfg_root + cfg_name
    file = open(path, 'r', encoding='utf-8')
    string = file.read()
    dict = yaml.safe_load(string)
    return dict



