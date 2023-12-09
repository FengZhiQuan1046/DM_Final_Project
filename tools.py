from copy import copy
import numpy as np
import pandas as pd
import random
import torch
import logging
from configparser import ConfigParser
import os
logger = logging.getLogger(__name__)

def parse_configs(configs_path: str) -> ConfigParser:
    parser = ConfigParser()
    parser.read(filenames=configs_path)
    return parser


def init_seeds(npSeed = 0, pdSeed = 0, randomSeed = 0, torchseed = 0):
    np.random.seed(seed=npSeed)
    pd.core.common.random_state(pdSeed)
    random.seed(a=randomSeed)
    torch.manual_seed(torchseed)
    logger.info(msg=f"Seeds: \n Numpy: {npSeed}\n Pandas: {pdSeed}\n Random: {randomSeed}\n Torch: {randomSeed}")

def get_logger(log_name: str, save_dir: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level=logging.DEBUG)
    sh.setFormatter(fmt=formatter)

    filename = os.path.join(save_dir, log_name)
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename=filename, mode="a", encoding="UTF-8")
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(fmt=formatter)

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(hdlr=sh)
    logger.addHandler(hdlr=fh)

    return logger