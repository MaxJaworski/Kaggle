import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('input/train.csv', index_col='id')

#split into features and target
X = df.drop(columns=['exam_score'])
y = df['exam_score']

#encode categorical variables with get_dummies
cols_to_encode = ['gender', 'course', 'internet_access', 'sleep_quality', 'study_method', 'facility_rating', 'exam_difficulty']
X = pd.get_dummies(X, columns=cols_to_encode, drop_first=True)

#standardise features
cols_to_standardise = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
scaler = StandardScaler()
X[cols_to_standardise] = scaler.fit_transform(X[cols_to_standardise])

#import and preprocess test data, align with X
test = pd.read_csv('input/test.csv', index_col='id')
test = pd.get_dummies(test, columns=cols_to_encode, drop_first=True)
test[cols_to_standardise] = scaler.transform(test[cols_to_standardise])
X, test = X.align(test, join='left', axis=1, fill_value=0)

#build mlp

model = Sequential()

model.add(Dense(units=256, activation='relu', input_shape=(X.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')

early_stop = EarlyStopping(monitor='val_rmse', mode='min', patience=10, restore_best_weights=True)

history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=1024, callbacks=[early_stop])

#make predictions
preds = model.predict(test).flatten()
preds = np.round(preds, 1)
preds = pd.DataFrame({'id': test.index, 'exam_score':preds})
preds.to_csv('results.csv', index=False)