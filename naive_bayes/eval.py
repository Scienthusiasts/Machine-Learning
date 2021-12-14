import numpy as np

import sys;sys.path.append('../metrics')
from metrics import metrics

pred_pos = np.load('imdb_nounique/pred_pos.npy')
pred_neg = np.load('imdb_nounique/pred_neg.npy')
pred_all = np.concatenate([pred_pos, ~pred_neg])
y = np.concatenate([np.ones(pred_pos.shape[0]), np.zeros(pred_neg.shape[0])])

pos_acc = sum(pred_pos) / pred_pos.shape[0]
neg_acc = sum(pred_neg) / pred_neg.shape[0]
print('pos acc:%f, neg acc:%f, total acc:%f' % (pos_acc, neg_acc, (pos_acc + neg_acc) / 2))

# 绘制混淆矩阵:
label = np.arange(2)
metrics.confusion_matrix_vis(y, pred_all, label)
metrics.precision_recall(y, pred_all, ['pos','neg'])