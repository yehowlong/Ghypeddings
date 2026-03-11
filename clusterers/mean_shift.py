from Ghypeddings.clusterers.utils import calculate_metrics

from sklearn.cluster import MeanShift

def mean_shift(X,y):
    y_pred = MeanShift().fit_predict(X)
    y_pred[y_pred>0] = -1
    y_pred[y_pred == 0] = 1
    y_pred[y_pred == -1]=0
    return calculate_metrics(y,y_pred)