from clusterers.utils import calculate_metrics
from sklearn.cluster import DBSCAN

def dbscan(X,y,eps=1e-4,min_samples=300):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X)
    mask = y_pred != -1
    y_true_filtered = y[mask]
    y_pred_filtered = y_pred[mask]
    y_pred_filtered[y_pred_filtered>0] = -1
    y_pred_filtered[y_pred_filtered == 0] = 1
    y_pred_filtered[y_pred_filtered == -1]=0
    return calculate_metrics(y_true_filtered,y_pred_filtered)