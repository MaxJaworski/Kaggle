import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

df = pd.read_csv('input/train.csv', index_col='id')

#split into features and target
X = df.drop(columns=['exam_score'])
y = df['exam_score']

#new features
X['new_1'] = X['study_hours']*X['class_attendance']
X['new_2'] = X['study_hours']*X['sleep_hours']
X['new_3'] = X['sleep_hours']*X['class_attendance']

#encode categorical variables with get_dummies
cols_to_encode = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)

#standardise features
cols_to_standardise = ['age', 'study_hours', 'class_attendance', 'sleep_hours', 'new_1', 'new_2', 'new_3']
scaler = StandardScaler()
X[cols_to_standardise] = scaler.fit_transform(X[cols_to_standardise])

#import and preprocess test data, align with X
test = pd.read_csv('input/test.csv', index_col='id')
test['new_1'] = test['study_hours']*test['class_attendance']
test['new_2'] = test['study_hours']*test['sleep_hours']
test['new_3'] = test['sleep_hours']*test['class_attendance']
test = pd.get_dummies(test, columns=cols_to_encode, drop_first=True)
test[cols_to_standardise] = scaler.transform(test[cols_to_standardise])
X, test = X.align(test, join='left', axis=1, fill_value=0)

#build model
xgb_model = XGBRegressor(
    n_estimators=1000,     
    max_depth=6,           
    learning_rate=0.05,    
    subsample=0.8,         
    colsample_bytree=0.8,  
    tree_method='hist',   
    eval_metric='rmse',    
    n_jobs=-1,             
    random_state=42
)
xgb_model.fit(X, y)

#make predictions
preds = xgb_model.predict(test)
preds = np.round(preds, 1)
preds = pd.DataFrame({'id': test.index, 'exam_score':preds})
preds.to_csv('results_3.csv', index=False)