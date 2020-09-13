# -*- encoding: utf-8 -*-
'''
@File    :   evaluation.py
@Time    :   2020/09/07 10:25:58
@Author  :   Xiaoxiao Ma 
@Version :   1.0
@Contact :   xiaoxiao.ma2@hdr.mq.edu.au
@License :   (C)Copyright 2020-2025, Xiaoxiao Ma
@Desc    :   Contact author for details
'''

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def evaluation(y_true, y_pred):
    
    result = {}
    
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    result.update({"f1" : f1})
    result.update({"auc" : roc_auc})
    
    return result

    