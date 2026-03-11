from sklearn.ensemble import RandomForestClassifier

def random_forest(X,y,seed,n_estimators=10,max_depth=10,max_features='log2'):
    clf = RandomForestClassifier(max_features=max_features,n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    return clf.fit(X, y)