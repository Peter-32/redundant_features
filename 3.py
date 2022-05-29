from numpy import random
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.datasets import fetch_california_housing
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
m, n = housing.data.shape
X_original = pd.DataFrame(housing.data)
ss = StandardScaler()
X_original.columns = housing.feature_names
y = housing.target

for k in [0,10,100,250,500,1000,3000]:
    X = X_original.copy()
    new_data = random.standard_normal((X.shape[0], k))
    X = pd.concat([X, pd.DataFrame(new_data)], axis='columns', ignore_index=True)

    seed(32)
    split_size = 0.4 if X.shape[0] < 100000 else 40000/X.shape[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)

    X_train = ss.fit_transform(X_train, y_train)
    X_test = ss.transform(X_test)

    lr = ElasticNet()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    if k == 0:
        first_mse = mse
    print(k, mse, (first_mse-mse) / first_mse)
