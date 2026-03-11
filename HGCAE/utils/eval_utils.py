from sklearn.metrics import average_precision_score, accuracy_score, f1_score

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(labels,preds)
    f1 = f1_score(labels,preds, average=average)
    return accuracy, f1

