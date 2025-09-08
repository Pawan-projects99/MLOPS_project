import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from dataclasses import dataclass
import os
import sys
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.tree import DecisionTreeRegressor
from src.utils import evaluate_model, save_obj
from sklearn.metrics import r2_score

class ModelTrainerConfig:
    model_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Linear Regression" : LinearRegression(),
                "Logistic Regression" : LogisticRegression(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "KNN" : KNeighborsRegressor(),
                "XG Boost" : XGBRegressor(),
                "Cat Boost" : CatBoostRegressor()
            }
            params = {
                "Linear Regression":{},
                "Logistic Regression":{
                    #'penalty':['l1','l2','elasticnet','none'], #Type of regularization applied:•	'l1' → Lasso regularization (sparsity, forces coefficients to zero).•	'l2' → Ridge regularization (shrinks coefficients, prevents overfitting).•	'elasticnet' → Combination of L1 + L2.
                    #'C' : np.logspace(-4,4,5), #Inverse of regularization strength.•	Smaller C → stronger regularization.•	Larger C → weaker regularization (model fits training data more closely).  Values like [1e-4, 1e-2, ..., 1e4] (5 evenly spaced values in log scale)
                    #'solver': ['lbfgs','newton-cg','liblinear','sag','saga'], #Algorithm used to optimize Logistic Regression:'lbfgs': Good default for small/medium datasets, supports only L2 or none •	'newton-cg': Supports L2 or none.•	'liblinear': Works well for small datasets, supports L1 and L2.•	'sag': Stochastic Average Gradient, efficient for large datasets, supports L2 or none.•	'saga': Supports all penalties (l1, l2, elasticnet, none), good for large datasets.
                    #'max_iter'  : [100,250,500] # Maximum number of iterations the solver runs before stopping.
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001], # 	Shrinks the contribution of each weak learner.•	Smaller values → slower learning but may improve generalization.•	Larger values → faster learning, but risk of overfitting.
                    'loss':['linear','square','exponential'],# Controls how errors are penalized when updating weights.•	'linear': Linear loss (default).•	'square': Squared loss (penalizes larger errors more heavily).•	'exponential': Exponential loss (weights big errors much more).
                    'n_estimators': [8,16,32,64,128,256]# Number of weak learners (usually decision stumps) to use in the ensemble.
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],# loss function to optimize
                    'learning_rate':[.1,.01,.05,.001],
                    #'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],# Fraction of samples used for fitting each base learner.
                    #'criterion':['squared_error', 'friedman_mse'], # for quality of splits
                    #'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2'],
                },
                "KNN":{
                    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
                    'weights': ['uniform', 'distance'], # weights → how neighbors contribute:•	"uniform" → all neighbors equal.•	"distance" → closer neighbors have more weight.
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    #'leaf_size': [10, 20, 30, 40, 50],
                    'p': [1, 2] # p → power parameter for Minkowski metric:•	p=1 → Manhattan distance (L1).•	p=2 → Euclidean distance (L2, default).•	p>2 → other Minkowski metrics.
                },
                "XG Boost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },   
                "Cat Boost":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                }            
            }
            logging.info("Evaluating the models using train and test data")
            model_report:dict = evaluate_model(
                X_train = X_train,
                y_train = y_train,
                X_test = X_test,
                y_test = y_test,
                models = models,
                params = params
                )
            print(model_report)
            # best_model_score = max(sorted(model_report.values()))
            # best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            # best_model = model_report[best_model_name]
            best_model_name = max(model_report, key = lambda name:model_report[name]["test_r2"]) 
            best_model_score = model_report[best_model_name]["test_r2"]
            best_model = models[best_model_name]
            
            if best_model_score<=0.6:
                raise CustomException("No best model found")
            logging.info("best model found")
            
            #saving the model
            save_obj(file_path=self.model_trainer_config.model_path, obj=best_model)
            logging.info("saved the model to the file path")

            #Making predictions using the best model
            predicted=best_model.predict(X_test)
            logging.info("predicted using the best model")
            
            #Calculating the r2_score
            r2_square = r2_score(y_test, predicted)
            logging.info("predicting the r2_score using the best model")
            print(f"best model is: {best_model_name} and best r2_score is: {best_model_score}")
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)