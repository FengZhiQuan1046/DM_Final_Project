from params import *
import sys
from model.catboost import initialize_model, train_rf
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser
from tools import *
import optuna
from params import init_params
from data.data4regressor import Data4Regressor
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    p = init_params()
    logging_name = f"logging_{p['version']}.log"

    model_configs = {
        'model_name' : p['model_name'],
        'config_dir' : os.path.join(p['config_dir'], 'CatBoostRegressor.json'),
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
            "max_depth" : trial.suggest_int('maxd', 12, 30, log=True), 
            "min_samples_split" : trial.suggest_int('mss', 2, 5, log=True), 
            "min_samples_leaf" : trial.suggest_int('msl', 8, 20, log=True), 
            "min_weight_fraction_leaf" : trial.suggest_float('mwfl', 0.00001, 0.5, log=True), 
            "max_leaf_nodes" : trial.suggest_int('mln', 8, 20, log=True), 
            "min_impurity_decrease" : trial.suggest_float('mid', 0.0000001, 1.0, log=True), 
            "n_jobs" : 16, 
            "verbose" : trial.suggest_float('vb', 0.0000001, 1.0, log=True)
        }

        for k in optim_params:
            model_params[k] = optim_params[k]

        model = initialize_model(configs=model_configs, model_params=model_params)
        model, loss = train_rf(model, train_set, val_set)
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

    