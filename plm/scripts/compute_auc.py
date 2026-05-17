import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn.functional as F


def compute_auc(gold, pred):
    y_true = np.array(gold)
    y_pred = np.array(pred)

    '''
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_binary = lb.transform(y_true)
    '''
    y_true_binary = F.one_hot(
        torch.from_numpy(y_true),
        num_classes=y_pred.shape[-1],
    ).numpy()

    if len(set(gold)) < y_pred.shape[-1]:
        y_pred = y_pred[:, y_true_binary.sum(0) > 0]
        y_true_binary = y_true_binary[:, y_true_binary.sum(0) > 0]

    auc = roc_auc_score(y_true_binary, y_pred, average="macro", multi_class="ovr")
    return auc
