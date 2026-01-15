import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('input/train.csv', index_col='id')

#split into features and target
X = df.drop(columns=['exam_score'])
y = df['exam_score']
    
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#select only most correlated
X_train_best = X_train[['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'study_method']]
X_test_best = X_test[['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'study_method']]

#encode categorical variables with get_dummies
cols_to_encode = ['sleep_quality', 'study_method']
X_train_best = pd.get_dummies(X_train_best, columns=cols_to_encode, drop_first=True)
X_test_best = pd.get_dummies(X_test_best, columns=cols_to_encode, drop_first=True)

#standardise features
cols_to_standardise = ['study_hours', 'class_attendance', 'sleep_hours']
scaler = StandardScaler()
X_train_best[cols_to_standardise] = scaler.fit_transform(X_train_best[cols_to_standardise])
X_test_best[cols_to_standardise] = scaler.transform(X_test_best[cols_to_standardise])

#parameter tuning

rfr = RandomForestRegressor()
rfr.fit(X_train_best, y_train)
print(rfr.score(X_test_best, y_test))

# param_grid = {
#     "n_estimators": [200, 500],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 4],
#     "max_features": ["sqrt", "log2"],
#     "bootstrap": [True]
# }

# rfrcv = GridSearchCV(rfr, param_grid=param_grid, cv=5)
# rfrcv.fit(X_train_best, y_train)
# print(rfrcv.best_params_)