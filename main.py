from params import *
import sys

from argparse import ArgumentParser, Namespace
from configparser import ConfigParser

import optuna
from data.data4classifier import data4classifier

if __name__ == "__main__":
    train_ds = data4classifier(train_file)
    # test_ds = data4classifier(test_file)
    