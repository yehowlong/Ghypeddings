from Ghypeddings.clusterers.utils import calculate_metrics

from sklearn.cluster import KMeans


def kmeans(X,y,n_clusters=2,n_init=10):
    model = KMeans(n_clusters=n_clusters,n_init=n_init)
    model.fit(X)
    y_pred = model.labels_
    y_pred[y_pred!=1]=0
    return calculate_metrics(y,y_pred)