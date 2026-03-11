from sklearn.neighbors import KNeighborsClassifier

def KNN(X,y,k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    return knn.fit(X, y)