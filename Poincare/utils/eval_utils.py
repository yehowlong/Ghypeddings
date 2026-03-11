from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score,roc_auc_score

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(labels,preds)
    recall = recall_score(labels,preds)
    precision = precision_score(labels,preds)
    roc_auc = roc_auc_score(labels,preds)
    f1 = f1_score(labels,preds, average=average)
    return accuracy, f1,recall,precision,roc_auc

