import os
import numpy as np

def GetSVDProto(r):
    filename = "./proto/SVD/cifar10_SVD%d.prototxt" % r
    if not os.path.exists(filename):
        fin = open("./proto/cifar10_SVD.template", "r")
        fout = open(filename, "w")
        for line in fin.readlines():
            fout.write(line.replace("$", str(r)))
        fin.close()
        fout.close()

    return filename

def get_comfusion_matrix(label, pre, k):
    # TP, FN
    # FP, TN
    wlabel = (label == k)
    wpre = (pre == k)
    TP = np.sum(wlabel & wpre)
    FN = np.sum(wlabel & ~wpre)
    FP = np.sum(~wlabel & wpre)
    TN = np.sum(~wlabel & ~wpre)
    return np.matrix([[TP, FN], [FP, TN]])

def eval_result(label, pre, k):
    cm = get_comfusion_matrix(label, pre, k) 
    print ("Confusion Matrix:")
    print (cm)
    acc = (cm[0,0] + cm[1,1]) * 1.0 / np.sum(cm)
    precision = cm[0,0] * 1.0 / np.sum(cm[:, 0])
    recall = cm[0,0] * 1.0 / np.sum(cm[0, :])
    F = 2.0 * recall * precision / (recall + precision)
    print ("Accuracy: %f" % acc)
    print ("Precision: %f" % precision)
    print ("Recall: %f" % recall)
    print ("F-measure: %f" % F)
