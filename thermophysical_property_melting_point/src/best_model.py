#after thorough analysis, xgboost provided the best results, with default params, SMILES removed, features standardised 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

#load train and test data
train = pd.read_csv(r'C:\Users\maxjj\Desktop\Kaggle\thermophysical_property_melting_point\input\train.csv', index_col='id')
test = pd.read_csv(r'C:\Users\maxjj\Desktop\Kaggle\thermophysical_property_melting_point\input\test.csv', index_col='id')

#split train into target and features, drop SMILES
X = train.drop(columns=['Tm', 'SMILES'])
y = train['Tm']
test = test.drop(columns=['SMILES'])
test = test.sort_index()
indices = test.index 

#standardise features
sc = StandardScaler()
X = sc.fit_transform(X)
test = sc.transform(test)

#build model
xgb = XGBRegressor()
xgb.fit(X, y)

#predictions
preds = xgb.predict(test)
preds = np.round(preds, 1)
preds = pd.DataFrame({'id': indices, 'Tm':preds})
preds.to_csv('results.csv', index=False)
