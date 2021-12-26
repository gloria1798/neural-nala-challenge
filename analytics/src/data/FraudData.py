from typing import Any, Dict
from pathlib import Path
import argparse
import ast
import json
import logging
import sys
import datetime
import csv
import pandas as pd
import itertools
from itertools import tee
from itertools import zip_longest as zip
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector

DATA_DIR = Path(__file__).resolve().parents[3]
START = 0
END = 22000
# DF = pd.read_csv(f'{DATA_DIR}/dataset/fraud_data_complete_final.csv', sep=";", encoding='utf8')
DF = pd.read_csv(f'{DATA_DIR}/dataset/fraud_data_complete_final.csv', encoding='utf8')

class FraudData():
	def __init__(self, args: argparse.Namespace = None):
		self.args = vars(args) if args is not None else {}
		self.start = self.args.get("start", START)
		self.end = self.args.get("end", END)
		self.data = DF		
		self.data.drop(self.data.columns[0], axis=1)
		y = self.data["fraude"]
		data_features = ['dispositivo', 'establecimiento','ciudad', 'tipo_tc', 'linea_tc','interes_tc', 'is_prime']
		X = self.data[data_features]
		self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(X, y, random_state = 0)
		numeric_cols = self.train_X.select_dtypes(include=['float64', 'int']).columns.to_list()
		cat_cols = self.train_X.select_dtypes(include=['object', 'category', 'bool']).columns.to_list()
		self.preprocessor = ColumnTransformer(
                   [('scale', StandardScaler(), numeric_cols),
		    ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
                remainder='passthrough')
	
	@ staticmethod
	def clean_data_query(data):
		ind = 1		
		data = {ind: data}				
		df = pd.DataFrame(data)
		data = df.transpose()		
		# data['dispositivo'][ind] = json.loads(data['dispositivo'][ind].replace(';', ','))['os']		
		# data['dispositivo'][ind] = data['dispositivo'][ind].replace(',', 'Unknown')
		# data['dispositivo'][ind] = data['dispositivo'][ind].replace('%%', 'Uknown')
		data['tipo_tc'][ind] = data['tipo_tc'][ind].replace('FÃsica', 'Fisica')		
		data['monto'][ind] = float(data['monto'][ind].replace(',', '.'))
		data['dcto'][ind] = float(str(data['dcto'][ind]).replace(',', '.'))
		data['cashback'][ind] = float(data['cashback'][ind].replace(',', '.'))		
		return data

	def data_preprocess(self,data):
  		a,b,c,d = self.data_after_split_and_preprocess()  		
  		return self.preprocessor.transform(data)

	def data_after_split_and_preprocess(self):
		train_X = self.preprocessor.fit_transform(self.train_X)
		val_X  = self.preprocessor.transform(self.val_X)
		return train_X, val_X, self.train_y, self.val_y

	def capture(self):
		self.minibatch = self.data[self.start:self.end]
		return self.minibatch

	@ classmethod
	def data_dirname(cls):
		return Path(__file__).resolve().parents[3] / "dataset"

	@ staticmethod
	def add_to_argparse(parser):
		parser.add_argument("--start", type=str,
							default=START, help="start.")
		parser.add_argument("--end", type=str,
							default=END, help="end.")
        

