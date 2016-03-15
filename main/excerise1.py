print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


def getResultByThreshold(input, threshold):
    return [1 if x > threshold else 0 for x in input]


true_label = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
classifier_a = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]
classifier_b = [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
classifier_c = [0.8, 0.9, 0.7, 0.6, 0.4, 0.8, 0.4, 0.4, 0.6, 0.4, 0.4, 0.4, 0.2]
threshold = [0.5, 0.6, 0.3]

# result = getResultByThreshold(classifier_c,threshold[2])
result = classifier_b

fpr, tpr, notused = roc_curve(true_label, result)
roc_auc = auc(fpr, tpr)
# Compute micro-average ROC curve and ROC area

##############################################################################
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
