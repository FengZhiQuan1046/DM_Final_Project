from copy import copy
import numpy as np
import pandas as pd
import random
import torch
import logging
from configparser import ConfigParser
logger = logging.getLogger(__name__)

def parse_configs(configs_path: str) -> ConfigParser:
    parser = ConfigParser()
    parser.read(filenames=configs_path)
    return parser


def init_seeds(npSeed = 0, pdSeed = 0, randomSeed = 0, torchseed = 0):
    np.random.seed(seed=npSeed)
    pd.core.common.random_state(pdSeed)
    random.seed(a=randomSeed)
    torch.seed(torchseed)
    logger.info(msg=f"Seeds: \n Numpy: {npSeed}\n Pandas: {pdSeed}\n Random: {randomSeed}\n Torch: {randomSeed}")