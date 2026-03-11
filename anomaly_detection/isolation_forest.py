from Ghypeddings.anomaly_detection.utils import calculate_metrics


from sklearn.ensemble import IsolationForest

def isolation_forest(X,y,anomalies_percentage = 0.1):
    model = IsolationForest(contamination=anomalies_percentage)
    model.fit(X)
    y_pred = model.predict(X)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1]= 1
    return calculate_metrics(y,y_pred)