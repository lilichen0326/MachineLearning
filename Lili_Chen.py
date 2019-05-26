import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

train_file = pd.read_csv('Admission_Predict.csv')
train_data = np.array(train_file)

train_x = train_data[:,1:7]
train_y = train_data[:,8]

svrlassifier = SVR(kernel='linear', gamma='auto')
svrlassifier.fit(train_x, train_y)

test_file = pd.read_csv('Admission_Predict_Ver1.1.csv')
test_data = np.array(test_file)
test_x = test_data[:,1:7]
test_y = test_data[:,8]

y_pred = svrlassifier.predict(test_x)

print('The score for the Support Vector Machine:', svrlassifier.score(test_x, test_y))

model = LinearRegression(fit_intercept= True)
model.fit(train_x, train_y)
y_pred_reg = model.predict(test_x)
print('The score for the Linear Regression:', model.score(test_x, test_y))

rf = RandomForestRegressor(n_estimators = 2000, random_state = 50)
rf.fit(train_x, train_y)
predictions = rf.predict(test_x)
print('The score for the Random Forest:', rf.score(test_x, test_y))
