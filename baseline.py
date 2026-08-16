import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

root = './data_house.csv'
df = pd.read_csv(root)
target = "MedHouseVal"

x = df.drop(target, axis=1)
y = df[target]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

num_tranformer = Pipeline([
  ("imputer", SimpleImputer(strategy="median")),
  ("scaler", StandardScaler())
])

colum_num = list(x.columns.values)
preprocessor = ColumnTransformer([
  ("num_feature", num_tranformer, colum_num)
])

reg = Pipeline([
  ("preprocessor", preprocessor),
  ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

params = {
  "model__n_estimators":[100,200,300],
  "model__criterion" :['squared_error', 'absolute_error', 'poisson']
}

grid_search = GridSearchCV(estimator=reg, param_grid=params, cv=4, verbose=2, scoring="r2")
grid_search.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)

print(grid_search.best_params_)
print("R²   :", r2_score(y_test, y_predict))
print("MAE  :", mean_absolute_error(y_test, y_predict))
print("MSE  :", mean_squared_error(y_test, y_predict))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_predict)))