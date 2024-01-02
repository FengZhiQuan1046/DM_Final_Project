import os
import random

import numpy as np
import pandas as pd
import sys

from catboost import CatBoostRegressor, Pool
from configparser import ConfigParser
from logging import getLogger
import json
from sklearn.metrics import mean_squared_error
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

logger = getLogger(__name__)

def get_model_parameters(model_params_path: str) -> dict:
    model_params = None

    with open(file=model_params_path, mode="r", encoding="UTF-8") as file:
        model_params = json.load(fp=file)

    return model_params

def initialize_model(configs: dict, model_params = None) -> CatBoostRegressor:
    models = {"CatBoostRegressor": CatBoostRegressor,
              "RandomForestRegressor": RandomForestRegressor,
              "LGBMRegressor": LGBMRegressor,
              "XGBRegressor": XGBRegressor}

    model_name = configs["model_name"]
    model_params_path = configs["config_dir"]
    # os.path.join("configs",
                                    #  (model_name + ".json"))
    # model_params = get_model_parameters(model_params_path=model_params_path)
    if model_params == None:
        with open(file=model_params_path, mode="r", encoding="UTF-8") as file:
            model_params = json.load(fp=file)
        model = models[model_name](**model_params)
    else: 
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




# def train(model, train_data, val_data=None):
#     fit_log = StringIO()
#     logger.info(msg=f"Training ...")

#     with redirect_stdout(new_target=fit_log), redirect_stderr(
#             new_target=fit_log):
#         model.fit(train_data, log_cout=sys.stdout, log_cerr=sys.stderr)
#         if val_data != None: 
#             # y = pd.DataFrame(val_data, columns=['label'])
#             loss = validate(model, val_data['data'], val_data['label'])
#         else: loss = model.get_best_score()

#     logger.info(msg=f"Training log: \n{fit_log.getvalue()}")
#     return model, loss

    
# def validate(model, x, y):
#     logger.info(msg=f"Validating ..")

#     unlabeled_data = x

#     prediction = model.predict(unlabeled_data)
    
#     rmse = mean_squared_error(y, prediction)
#     logger.info(msg=f"Validate finished.")
#     return rmse



# def inference(model, data, keys, save_dir):
#     logger.info(msg=f"Inferencing ...")

#     # model = parameters["model"]
#     unlabeled_data = data

#     prediction = model.predict(unlabeled_data)

#     df = keys
#     df = pd.concat(objs=[df, pd.Series(data=prediction, name="label")], axis=1)

#     df.to_csv(path_or_buf=save_dir, index=False)

#     logger.info(msg=f"Number 1s: {np.count_nonzero(a = prediction == 1)}")



# def train_rf(model, train, val_data=None):
#     fit_log = StringIO()
#     logger.info(msg=f"Training ...")
#     with redirect_stdout(new_target=fit_log), redirect_stderr(
#             new_target=fit_log):
#         model.fit(train['data'], train['label'])
#         if val_data != None: 
#             # y = pd.DataFrame(val_data, columns=['label'])
#             loss = validate(model, val_data['data'], val_data['label'])
#         else: loss = model.get_best_score()

#     logger.info(msg=f"Training log: \n{fit_log.getvalue()}")
#     return model, loss

    
# def validate_rf(model, x, y):
#     logger.info(msg=f"Validating ..")

#     unlabeled_data = x

#     prediction = model.predict(data=unlabeled_data)
    
#     rmse = mean_squared_error(y, prediction)
#     logger.info(msg=f"Validate finished.")
#     return rmse



# def inference_rf(model, data, keys, save_dir):
#     logger.info(msg=f"Inferencing ...")

#     # model = parameters["model"]
#     unlabeled_data = data

#     prediction = model.predict(data=unlabeled_data)

#     df = keys
#     df = pd.concat(objs=[df, pd.Series(data=prediction, name="label")], axis=1)

#     df.to_csv(path_or_buf=save_dir, index=False)

#     logger.info(msg=f"Number 1s: {np.count_nonzero(a = prediction == 1)}")


