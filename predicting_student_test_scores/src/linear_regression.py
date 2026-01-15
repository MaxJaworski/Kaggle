import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import cross_val_score, train_test_split
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

# #build, train, and evaluate lr model 
# lr = LinearRegression()
# lr.fit(X_train_best, y_train)
# print(lr.score(X_test_best, y_test))

# #build, train, and evaluate lasso model
# lasso = Lasso(alpha=0.014419578466425472)
# lasso.fit(X_train_best, y_train)
# print(lasso.score(X_test_best, y_test))

#build, train, and evaluate ridge model
ridge = RidgeCV(alphas=[0.001,0.01,0.1,1,10,100])
ridge.fit(X_train_best, y_train)
#print(ridge.score(X_test_best, y_test))

# #elastic net
# en = ElasticNetCV(cv=10)
# en.fit(X_train_best, y_train)
# print(en.score(X_test_best, y_test))

#ridgecv best of the bunch with 0.7545053039679808

#joblib.dump(ridge, 'best_model')

#import and preprocess test data, use model to make predictions
test = pd.read_csv('input/test.csv', index_col='id')

test_new = test[['study_hours', 'class_attendance', 'sleep_hours', 'sleep_quality', 'study_method']]
test_new = pd.get_dummies(test_new, columns=cols_to_encode, drop_first=True)
test_new[cols_to_standardise] = scaler.transform(test_new[cols_to_standardise])

preds = np.round(ridge.predict(test_new), 1)

preds = pd.DataFrame({'id': test.index, 'exam_score':preds})

preds.to_csv('results.csv', index=False)

