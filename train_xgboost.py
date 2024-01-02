from params import *
import sys
from model.catboost import initialize_model
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from tools import *
import optuna
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from params import init_params
from data.data4regressor import Data4Regressor
import pickle as pkl
from model.trainers import *

if __name__ == "__main__":
    p = init_params()
    logging_name = f"logging_{p['version']}.log"

    model_configs = {
        'model_name' : p['model_name'],
        'config_dir' : os.path.join(p['config_dir'], 'XGBRegressor.json'),
        'checkpoint_version' : 'None'
    }

    logger = get_logger(logging_name, p['loggings_root'])
    init_seeds()

    ds = Data4Regressor(p['train_file'])
    train_set, val_set = ds.split_xy()
    # model = initialize_model(model_configs)
    with open(os.path.join(p['config_dir'], f"{p['model_name']}.json"), 'r') as f:
        model_params = json.load(f)
    
    checkpoint_save_dir = f"{p['checkpoints_root']}/{p['version']}"
    if not os.path.exists(checkpoint_save_dir):
        os.mkdir(checkpoint_save_dir)
    trial_counter = 0
    configs_counter = 0
    min_rmse = 10.
    def opt(trial):
        global min_rmse, configs_counter, trial_counter
        optim_params = {
            "max_depth": trial.suggest_int("max_depth", 3, 15, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.99, log=True),
            "gamma": trial.suggest_float("gamma", 1e-7, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-7, 0.3, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-7, 0.1, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-7, 0.1, log=True),
            "num_parallel_tree": trial.suggest_int("num_parallel_tree", 3, 10, log=True)
        }

        for k in optim_params:
            model_params[k] = optim_params[k]

        # model_params["feature_weights"] = {
        #                     'id': 1, 
        #                     'words_count': trial.suggest_float("words_count", 0.01, 10.0, log=True), 
        #                     'remove_amount': trial.suggest_float("remove_amount", 0.01, 10.0, log=True), 
        #                     'copy_amount': trial.suggest_float("copy_amount", 0.01, 10.0, log=True), 
        #                     'average_sentence_len': trial.suggest_float("average_sentence_len", 0.01, 10.0, log=True), 
        #                     'paragraph_num':trial.suggest_float("paragraph_num", 0.01, 10.0, log=True), 
        #                     'edit_time': trial.suggest_float("edit_time", 0.01, 10.0, log=True), 
        #                     'audio_time': trial.suggest_float("audio_time", 0.01, 10.0, log=True), 
        #                     'media_time': trial.suggest_float("media_time", 0.01, 10.0, log=True), 
        #                     'stop_rate': trial.suggest_float("stop_rate", 0.01, 10.0, log=True), 
        #                     'max_mark_patterns_len': trial.suggest_float("max_mark_patterns_len", 0.01, 10.0, log=True)
        #                 }

        model = initialize_model(configs=model_configs, model_params=model_params)
        model, loss = train_lg(model, train_set, val_set)
        trial_counter += 1
        # logger.info(msg=f"================================>>>>>  loss: {loss} <<<<<================================")
        # print(loss)
        rmse = loss#['learn']['F1:use_weights=false']
        if min_rmse > rmse:
            min_rmse = rmse
            config_2b_saved = copy(model_params)
            config_2b_saved['rmse'] = rmse
            with open(os.path.join(checkpoint_save_dir, f"config_{trial_counter}_{configs_counter}.json"), 'w') as f:
                json.dump(config_2b_saved, f)
            configs_counter += 1
            with open(os.path.join(checkpoint_save_dir, f"model_{trial_counter}_{configs_counter}.mod"), 'wb') as f:
                pkl.dump(model, f)
        return rmse
    study = optuna.create_study(direction="minimize", study_name="autoctb")
    study.optimize(opt, n_trials=300)
    # best_trial = study.best_trial()

    