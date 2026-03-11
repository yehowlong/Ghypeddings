from sklearn.naive_bayes import GaussianNB

def naive_bayes(X,y):
    clf = GaussianNB()
    return clf.fit(X, y)