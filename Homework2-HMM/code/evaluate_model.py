# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2021/11/25
File   :evaluate_model.py
"""
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_model(test_y, y_pred):
    # calculate acc / f1 / recall
    print('Micro precision : ', precision_score(test_y, y_pred, average='micro'))
    print('Micro recall : ', recall_score(test_y, y_pred, average='micro'))
    print('Micro f1-score : ', f1_score(test_y, y_pred, average='micro'))
    print("\nEach class ROC")

    # calculate auc
    n_classes = 11
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print("\nEach class ROC :")
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print('class %d ROC : %d', i + 1, roc_auc[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(test_y.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print("\nAUC : ", roc_auc["micro"])
