import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import time

dataset = pd.read_csv('new_appdata10.csv')

### Data Preprocessing ###
response = dataset['enrolled']
dataset = dataset.drop(columns = 'enrolled')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset, response, test_size = 0.2, random_state = 0)

train_identifier = x_train['user']
test_identifier = x_test['user']
x_train = x_train.drop(columns = 'user')
x_test = x_test.drop(columns = 'user')

## Feature Scaling ##
# When we use Standard Scaler, the indices and the column names are lost.
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train2 = pd.DataFrame(sc_x.fit_transform(x_train))
x_test2 = pd.DataFrame(sc_x.transform(x_test))
x_train2.columns = x_train.columns.values
x_test2.columns = x_test.columns.values
x_train2.index = x_train.index.values
x_test2.index = x_test.index.values
x_train = x_train2
x_test = x_test2


### Model Building ###
from sklearn.linear_model import LogisticRegression
cls = LogisticRegression(random_state = 0, penalty = "l1", solver = 'saga')
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, fmt='g')
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = cls, X = x_train, y = y_train, cv = 10)
print("Logistic Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing Coefficients
pd.concat([
    pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["Features"]),
    pd.DataFrame(np.transpose(cls.coef_), columns = ["Coefficients"])
],axis = 1)


### Model Tuning ###

## Grid Search (Round 1) ##
from sklearn.model_selection import GridSearchCV

# Select Regularization Method
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
# Combine Parameters
params = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = cls, param_grid = params, scoring = "accuracy", cv = 10, n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
print(rf_best_accuracy, rf_best_parameters)

## Grid Search (Round 2)

# Select Regularization Method
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = [0.1, 0.5, 0.9, 1, 2, 5]
# Combine Parameters
params = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = cls, param_grid = params, scoring = "accuracy", cv = 10, n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(x_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
print(rf_best_accuracy, rf_best_parameters)
print(grid_search.best_score_)


### End of Model ###
# Formatting the Final Results
final_results = pd.concat([y_test, test_identifier], axis = 1).dropna()
final_results['predicted_results'] = y_pred
final_results[['user', 'enrolled', 'predicted_results']].reset_index(drop = True)








