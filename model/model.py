import os
import random

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool
from configparser import ConfigParser
from logging import getLogger
import json

logger = getLogger(__name__)

def get_model_parameters(model_params_path: str) -> dict:
    model_params = None

    with open(file=model_params_path, mode="r", encoding="UTF-8") as file:
        model_params = json.load(fp=file)

    return model_params

def initialize_model(configs: dict) -> CatBoostRegressor:
    models = {"CatBoostRegressor": CatBoostRegressor}

    model_name = configs["model_name"]
    model_params_path = os.path.join("configs",
                                     (model_name + ".json"))
    model_params = get_model_parameters(model_params_path=model_params_path)
    model = models[model_name](**model_params)

    if configs['checkpoint_version'] != "None":
        checkpoint_path = os.path.join(
            "outputs", "checkpoint", f"{configs['checkpoint_version']}.cbm")
        model.load_model(fname=checkpoint_path)

        logger.info(msg=f"Checkpoint has been loaded from {checkpoint_path}.")
    else:
        logger.info(msg=f"No checkpoint has been loaded.")

    logger.info(msg=f"The details of {model_name}: \n{model.get_params()}")
    logger.info(msg=f"{model_name} has been initialized.")

    return model