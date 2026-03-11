from Ghypeddings.anomaly_detection.utils import calculate_metrics


from sklearn.svm import OneClassSVM

def one_class_svm(X,y, kernel='rbf',nu=0.1):
    model = OneClassSVM(kernel=kernel, nu=nu)
    model.fit(X)
    y_pred = model.predict(X)
    y_pred[y_pred == -1]=0
    return calculate_metrics(y,y_pred)