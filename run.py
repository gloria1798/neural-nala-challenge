import pandas as pd
import numpy
import string
import sklearn.decomposition
import math
import statistics
import sklearn
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, completeness_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import svm
import os

import time
import sys
import json
import multiprocessing
import argparse
import importlib
import logging
# from mixpanel import Mixpanel
# from dotenv import dotenv_values

from analytics.src.data import FraudData
from analytics.src.models import RandomForest

# codeable.pem is generated in this dicrectory:
# cd /home/serendipita/Documents/python/deployAppUbuntuAWS
# ssh -i "codeable.pem" ubuntu@ec2-3-23-86-230.us-east-2.compute.amazonaws.com


def _import_class(module_and_clas_name: str) -> type:
    """Import class from a module, e.g. 'cryptomarket_realtime_analytics.src.data'"""
    module_name, class_name = module_and_clas_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_
  
def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_class", type=str, default="FraudData")
    parser.add_argument("--model_class", type=str, default="RandomForest")
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(
        f"analytics.src.data.{temp_args.data_class}")
    model_class = _import_class(
        f"analytics.src.models.{temp_args.model_class}")
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(
        f"analytics.src.data.{args.data_class}")
    model_class = _import_class(
        f"analytics.src.models.{args.model_class}")
    data = data_class(args=args)    
    model = model_class(args=args)
    train_X, val_X, train_y, val_y = data.data_after_split_and_preprocess()
    model.train(train_X, train_y)
    model.test(val_X, val_y)  
    return

if __name__ == '__main__': 
    main()

