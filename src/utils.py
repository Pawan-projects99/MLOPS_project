import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for name, model in models.items():
            if params[name]:
                grid = GridSearchCV(estimator=model, param_grid=params[name], cv = 5, scoring = r2_score, n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                # df = pd.DataFrame(grid.cv_results_)
                # print(df)
            else:
                # No tuning needed for LinearRegression
                best_model = model
                best_model.fit(X_train, y_train)
            
            #predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            
            #Evaluate
            train_model_mse_score = mean_squared_error(y_train, y_train_pred)
            train_model_r2_score = r2_score(y_train, y_train_pred)
            test_model_mse_score = mean_squared_error(y_test, y_test_pred)
            test_model_r2_score = r2_score(y_test,y_test_pred)
            
            #generating the report object
            report[name] = {
                "train_mse" : train_model_mse_score,
                "train_r2" : train_model_r2_score,
                "test_mse" : test_model_mse_score,
                "test_r2" : test_model_r2_score
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)