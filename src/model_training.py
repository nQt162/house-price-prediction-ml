'''
Nhiệm vụ:
  Tạo model
  Tạo Pipeline
  Xây dựng param_grid
  GridSearchCV
  Train
  Lưu best model
'''
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def create_model(preprocessor):
    """
    Tạo RandomForestRegressor và bọc vào Pipeline
    """
    model = RandomForestRegressor(random_state=42, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def create_param_grid():
    """
    Xây dựng param_grid (prefix model__ vì model nằm trong Pipeline)
    """
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__criterion": ["squared_error", "absolute_error", "poisson"],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }
    return param_grid


def train_model(pipeline, param_grid, X_train, y_train):
    """
    GridSearchCV: cv=4, scoring=r2, n_jobs=-1, fit, trả về best estimator
    """
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=4,
        scoring="r2",
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    print("Best params:", grid_search.best_params_)
    print("Best CV R²  :", grid_search.best_score_)

    return grid_search.best_estimator_


def save_model(model, path):
    """
    Dùng joblib để lưu model (tạo thư mục nếu chưa có)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print("Model đã lưu tại:", path)