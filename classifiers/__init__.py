from Ghypeddings.classifiers.svm import SVM
from Ghypeddings.classifiers.mlp import mlp
from Ghypeddings.classifiers.decision_tree import decision_tree
from Ghypeddings.classifiers.random_forest import random_forest
from Ghypeddings.classifiers.adaboost import adaboost
from Ghypeddings.classifiers.knn import KNN
from Ghypeddings.classifiers.naive_bayes import naive_bayes

from sklearn.metrics import accuracy_score , f1_score , recall_score , precision_score , roc_auc_score


def calculate_metrics(clf,X,y):
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    return accuracy,f1,recall,precision,roc_auc