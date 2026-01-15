import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

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
param_grid = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30]
}

rfr = RandomForestRegressor()
rfrcv = GridSearchCV(rfr, param_grid=param_grid, cv=5)
rfrcv.fit(X_train_best, y_train)
best_model = rfrcv.best_estimator_

#import and preprocess test data, use model to make predictions
test = pd.read_csv('input/test.csv', index_col='id')

test_new = test[['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'study_method']]
test_new = pd.get_dummies(test_new, columns=cols_to_encode, drop_first=True)
test_new[cols_to_standardise] = scaler.transform(test_new[cols_to_standardise])

preds = np.round(best_model.predict(test_new), 1)

preds = pd.DataFrame({'id': test.index, 'exam_score':preds})

preds.to_csv('results.csv', index=False)

