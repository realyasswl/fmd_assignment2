print(__doc__)
from operator import itemgetter
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
threshold = [0.5, 0.6, 0.1]

# result = getResultByThreshold(classifier_c, threshold[0])
result = classifier_b

fpr, tpr, notused = roc_curve(true_label, result)
# print(tpr)
# apply the CFN and CFP to our result, which only affects tpr
new_tpr = [0 if x == 0 else 1 / (1 + 5 * (-1 + 1 / x)) for x in tpr]
roc_auc = auc(fpr, new_tpr)

temp = 0
cls = 0
fprall = []
tprall = []
call = []
clslist = [classifier_a, classifier_b, getResultByThreshold(classifier_c, threshold[0])]
for classifier in range(len(clslist)):
    fpri, tpri, notused = roc_curve(true_label, clslist[classifier])
    new_tpri = [0 if x == 0 else 1 / (1 + 5 * (-1 + 1 / x)) for x in tpri]
    for j in range(len(fpri)):
        fprall.append(fpri[j])
        tprall.append(new_tpri[j])
        call.append(classifier)
        x = (new_tpri[j] - fpri[j] + 1) / 2
        if temp < x:
            temp = x
            cls = classifier
print(cls, temp)

f, t, c = (list(t) for t in zip(*sorted(zip(fprall, tprall, call))))

##############################################################################
# Plot of a ROC curve for a specific class
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(f, t, label='ROC curve (area = %0.2f)' % roc_auc)
clist=["a","b","c"]
for xy in zip(zip(f, t), c):
    ax.annotate('(%s)' % clist[xy[1]], xy=xy[0], textcoords='data')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
