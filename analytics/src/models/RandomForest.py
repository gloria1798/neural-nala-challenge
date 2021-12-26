from typing import Any, Dict
from pathlib import Path
import argparse
import json
import logging
import sys
import datetime
import csv
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix

DATA_DIR = Path(__file__).resolve().parents[3]
WEIGHTS_PATH = f"{DATA_DIR}/weights/finalized_model.sav"
NUM_ESTIMATORS = 100
RANDOM_STATE = 0

class RandomForest():
	def __init__(self, args: argparse.Namespace = None):	
		self.args = vars(args) if args is not None else {}
		self.num_estimators = self.args.get("num_estimators", NUM_ESTIMATORS)
		self.random_state = self.args.get("random_state", RANDOM_STATE)		
		self.model = RandomForestRegressor(n_estimators=self.num_estimators, random_state=self.random_state)
		self.filename = WEIGHTS_PATH

	# @staticmethod
	# def preprocess(train_X_):
	# 	numeric_cols = train_X.select_dtypes(include=['float64', 'int']).columns.to_list()
	# 	cat_cols = train_X.select_dtypes(include=['object', 'category', 'bool']).columns.to_list()
	# 	preprocessor = ColumnTransformer(
	# 					[('scale', StandardScaler(), numeric_cols),
	# 				('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
	# 					remainder='passthrough')
	# 	train_X_prep = preprocessor.fit_transform(train_X)
  	# 	return preprocessor

	def train(self, train_X_prep, train_y):
		self.model.fit(train_X_prep, train_y)
		pickle.dump(self.model, open(WEIGHTS_PATH, 'wb'))  		

	def test(self, val_X_prep, val_y):
		pred = self.model.predict(val_X_prep)
		metrics = self.metrics(val_y, pred)
		print(metrics)
	
	def predict(self, data):								
		loaded_model = pickle.load(open(self.filename, 'rb'))
		preds = loaded_model.predict(data)
		pred_test = round(pd.DataFrame(preds))
		if pred_test[0][0] == 0:
			pred_test = False
		else:
			pred_test = True
		return pred_test

	def metrics(self, val_y,preds):
		print("Mean Absolute Error:") 
		print(mean_absolute_error(val_y, preds))
		preds = pd.DataFrame(preds)
		preds = round(preds, 0)
		preds = preds.replace(0, False)
		preds = preds.replace(1, True)
		preds	
		confMat_RFM = confusion_matrix(val_y, preds)
		confMatList_RFM = confMat_RFM.tolist()
		TN = confMatList_RFM[0][0]
		TP = confMatList_RFM[1][1]
		FN = confMatList_RFM[1][0]
		FP = confMatList_RFM[0][1]
		accuracy_RFM = (TP + TN)/(TP + FP + FN + TN)
		recall_RFM = (TP) / (TP + FN)
		print("Accuracy:", accuracy_RFM)
		print("-"*50)
		if (TP + FP) == 0:
			print("Durante la prediccion no detectó fraudes a pesar de su existencia. No es posible calcular la precision")
		else:
			precision_RFM = (TP)/(TP + FP)
			print("Precision: ", precision_RFM)
			print("-"*50)
			print("Recall:", recall_RFM)
			print("-"*50)
			confMat_RFM
	
	@ staticmethod
	def add_to_argparse(parser):
		parser.add_argument("--num_estimators", type=int,
							default=NUM_ESTIMATORS, help="number of estimators.")
		parser.add_argument("--random_state", type=int,
							default=RANDOM_STATE, help="the random state.")