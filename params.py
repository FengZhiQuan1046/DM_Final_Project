from tools import parse_configs
# import logging
import os
# from scarf.model import SCARF
import json

def init_params():
    configs = parse_configs(configs_path='./configs/general.ini')
    output_root = configs.get(section="CODE", option="output_dir")
    dataset_dir = configs.get(section="CODE", option="dataset_dir")
    return {
        'seed' : configs.getint(section="CODE", option="seed"),
        'version' : configs.getint(section="CODE", option="version"),
        'output_root' : output_root,
        'config_dir' : configs.get(section="CODE", option="config_dir"),
        'checkpoints_root' : os.path.join(output_root, "checkpoints"),
        'loggings_root' : os.path.join(output_root, "loggings"),
        'medium_root' : os.path.join(output_root, "medium"),
        'predictions_root' : os.path.join(output_root, "predictions"),
        'model_name' : configs.get(section="MODEL", option="model_name"),
        'dataset_dir' : configs.get(section="CODE", option="dataset_dir"),
        'train_file' : os.path.join(dataset_dir, "features.json")
    }
