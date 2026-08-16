'''
Nhiệm vụ:
  Tạo model
  Tạo Pipeline
  Xây dựng param_grid
  GridSearchCV
  Train
  Lưu best model
'''
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def create_model(preprocessor):
    """
    TODO:
    Tạo RandomForestRegressor

    random_state=42
    n_jobs=-1

    Sau đó tạo Pipeline:
        preprocessor
        model
    """
    pass


def create_param_grid():
    """
    TODO:
    Xây dựng param_grid

    Ví dụ:
        n_estimators
        criterion
        max_depth
        min_samples_split
        min_samples_leaf
    """
    pass


def train_model(pipeline, param_grid, X_train, y_train):
    """
    TODO:
    - GridSearchCV
    - cv=4
    - scoring="r2"
    - n_jobs=-1
    - fit()
    """
    pass


def save_model(model, path):
    """
    TODO:
    Dùng joblib để lưu model
    """
    pass