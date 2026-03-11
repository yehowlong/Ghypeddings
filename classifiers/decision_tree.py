from sklearn.tree import DecisionTreeClassifier

def decision_tree(X,y,max_depth=2):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    return clf.fit(X, y)