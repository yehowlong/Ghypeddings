from sklearn.cluster import DBSCAN
from Ghypeddings.anomaly_detection.utils import calculate_metrics


def dbscan(X,y):
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    outliers = labels == -1
    return calculate_metrics(y,outliers)
