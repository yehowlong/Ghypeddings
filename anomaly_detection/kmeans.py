from sklearn.cluster import KMeans
from anomaly_detection.utils import calculate_metrics
import numpy as np

def kmeans(X,y,n_clusters,outlier_percentage=.1):
    model = KMeans(n_clusters=n_clusters)
    model.fit(X)
    # y_pred = model.predict(X)
    distances = model.transform(X).min(axis=1)
    threshold = np.percentile(distances, 100 * (1 - outlier_percentage))
    outliers = distances > threshold
    return calculate_metrics(y,outliers)