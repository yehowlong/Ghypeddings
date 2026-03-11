from sklearn import svm


def SVM(X,y,kernel='rbf',gamma='scale',C=1):    
    cls = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    return cls.fit(X, y)