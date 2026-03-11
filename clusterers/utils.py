## external evaluation metrics
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score
## additional evaluation metrics
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
## classification metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def calculate_metrics(y_true,y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    fmi = fowlkes_mallows_score(y_true, y_pred)
    homogeneity = homogeneity_score(y_true, y_pred)
    completeness = completeness_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)
    acc = accuracy_score(y_true,y_pred)
    f1 = f1_score(y_true,y_pred)
    rec = recall_score(y_true,y_pred)
    pre = precision_score(y_true,y_pred)
    roc = roc_auc_score(y_true,y_pred)
    return ari,nmi,fmi,homogeneity,completeness,v_measure,acc,f1,rec,pre,roc