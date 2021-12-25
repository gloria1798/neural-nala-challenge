import pandas as pd
import numpy
import string
from sklearn.model_selection import train_test_split
import sklearn.decomposition
import matplotlib.pyplot as plt
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
import graphviz
import pydot
import pydotplus
from sklearn import svm
import os
import seaborn as sns

import sys
import json
import multiprocessing
import argparse
import importlib
import logging
# from mixpanel import Mixpanel
# from dotenv import dotenv_values

from analytics.src.data import DBManager
from analytics.src.models import DBpreprocessing

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
    parser.add_argument("--data_class", type=str, default="TopicUserquote")
    parser.add_argument("--model_class", type=str, default="FraudDetection")
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(
        f"analytics.src.data.{temp_args.data_class}")
    model_class = _import_class(
        f"analytics.src.models.{temp_args.model_class}")
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)
    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)

    args = parser.parse_args()
    start = str(
        args.start_timestamp) if args.start_timestamp != 0 else "beginning"
    end = str(args.end_timestamp) if args.end_timestamp != int(
        sys.maxsize) else "now"
    logging.basicConfig(filename=f'debug_{temp_args.data_class}_{start}_{end}.log', level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logging.info('Start project')
    return parser


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    data_class = _import_class(
        f"analytics.src.data.{args.data_class}")
    model_class = _import_class(
        f"analytics.src.models.{args.model_class}")
    data = data_class(args)
    model = model_class(args=args)


    
    #logging.info(data_event_topic.info())
    try:
        while True:            
            #list_data = data_event_topic.capture()
            # if len(list_data) != 0:                
            #     data_class.toMixpanel(list_data, mp, SECRET)                
            # print(".")
            # print(json.dumps(list_data, indent=4, sort_keys=True))
    finally:
        #data_event_topic.close()

    return

if __name__ == '__main__': 
    main()

