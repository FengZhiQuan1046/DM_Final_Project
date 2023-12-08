from tools import parse_configs, init_seeds
# import logging
import os
from scarf.model import SCARF
import json

# configs_path = './configs/configs.ini'
configs = parse_configs(configs_path='./configs/general.ini')
seed = configs.getint(section="CODE", option="seed")
# init_seeds(,configs.getint(section="CODE", option="seed"),configs.getint(section="CODE", option="seed"),configs.getint(section="CODE", option="seed"))
output_root = configs.get(section="CODE", option="output_dir")
config_dir = configs.get(section="CODE", option="config_dir")
checkpoints_root = os.path.join(output_root, "checkpoints")
loggings_root = os.path.join(output_root, "loggings")
medium_root = os.path.join(output_root, "medium")
predictions_root = os.path.join(output_root, "predictions")

model_name = configs.get(section="MODEL", option="model_name")
# model_def = {
#     "scarf": SCARF
# }[model_name]

# with open(os.path.join(config_dir, f'{model_name}.json'), 'r') as f:
#     model_params = json.load(f)

dataset_dir = configs.get(section="CODE", option="dataset_dir")
train_file = os.path.join(dataset_dir, "features.json")
# train_scores = os.path.join(dataset_dir, "train_scores.csv")
# test_file = os.path.join(dataset_dir, "test_logs.csv")
