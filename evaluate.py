import numpy as np
import sys
from base import *

if len(sys.argv) == 1:
    PREDICT_FILENAME = "result/net_normal.npy"
else:
    PREDICT_FILENAME = sys.argv[1]

print ("Compare with Real Label and %s" % PREDICT_FILENAME)

num_kinds = 10
pre   = np.load(PREDICT_FILENAME)
label = np.load("label.npy")
n = len(label)

eval_result(label, pre, num_kinds)
