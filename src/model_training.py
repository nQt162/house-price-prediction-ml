'''
Nhiệm vụ:
  Predict test set
  Tính metric
  Phân tích kết quả
  Có thể vẽ biểu đồ
'''
import numpy as np
from sklearn.metrics import  r2_score,  mean_absolute_error, mean_squared_error


def predict(model, X_test):
    """
    TODO:
    model.predict(X_test)
    """
    pass


def evaluate_model(y_test, y_predict):
    """
    TODO:
    Tính:

    R²
    MAE
    MSE
    RMSE

    Trả về dictionary
    """
    pass


def print_metrics(metrics):
    """
    TODO:
    In kết quả đẹp
    """
    pass


def save_results(metrics, path):
    """
    TODO:
    Lưu kết quả vào CSV
    """
    pass