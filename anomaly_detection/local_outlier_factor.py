from sklearn.neighbors import LocalOutlierFactor
from anomaly_detection.utils import calculate_metrics
import numpy as np

def local_outlier_factor(X,y,n_neighbors=20,outlier_percentage=.1):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=outlier_percentage)
    y_pred = lof.fit_predict(X)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return calculate_metrics(y,y_pred)