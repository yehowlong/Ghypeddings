from sklearn.ensemble import AdaBoostClassifier

def adaboost(X,y,seed,n_estimators=2):
    ada_boost = AdaBoostClassifier(n_estimators=n_estimators, random_state=seed)
    return ada_boost.fit(X, y)