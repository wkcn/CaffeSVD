#coding=utf-8
# decompress ip2 layer
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import lmdb
import numpy as np
import os
import sys
from numpy import linalg as la
import matplotlib.pyplot as plt 
from base import *

CAFFE_HOME = "/opt/caffe/"
RESULT_DIR = "./result/"

SVD_R = 8
deploySVD = GetSVDProto(SVD_R)

deploy = "./proto/cifar10_quick.prototxt"
caffe_model = CAFFE_HOME + "/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5" 
train_db = CAFFE_HOME + "examples/cifar10/cifar10_train_lmdb"
test_db = CAFFE_HOME + "examples/cifar10/cifar10_test_lmdb"
mean_proto = CAFFE_HOME + "examples/cifar10/mean.binaryproto"
mean_npy = "./mean.npy"
mean_pic = np.load(mean_npy)

def read_db(db_name):
    lmdb_env = lmdb.open(db_name)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    datum = caffe.proto.caffe_pb2.Datum()

    X = []
    y = []
    cnts = {}
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = datum.label
        data = caffe.io.datum_to_array(datum)
        #data = data.swapaxes(0, 2).swapaxes(0, 1)
        X.append(data)
        y.append(label)
        if label not in cnts:
            cnts[label] = 0
        cnts[label] += 1
        #plt.imshow(data)
        #plt.show()
    return X, np.array(y), cnts

testX, testy, cnts = read_db(test_db)
#testX, testy, cnts = read_db(train_db)
print ("#train set: ", len(testX))
print ("the size of sample:", testX[0].shape)
print ("kinds: ", cnts)

if not os.path.exists("label.npy"):
    np.save("label.npy", testy)

# Load model and network
net = caffe.Net(deploy, caffe_model, caffe.TEST) 
for layer_name, param in net.params.items():
    # 0 is weight, 1 is biase
    print (layer_name, param[0].data.shape,net.blobs[layer_name].data.shape)

if SVD_R > 0:
    netSVD = caffe.Net(deploySVD, caffe_model, caffe.TEST)
    print ("SVD NET:")
    for layer_name, param in netSVD.params.items():
        # 0 is weight, 1 is biase
        print (layer_name, param[0].data.shape)

print (type(net.params))
print (net.params.keys())
print ("layer ip2:")
print ("WEIGHT:")
print (net.params["ip2"][0].data.shape)
print ("BIASES:")
print (net.params["ip2"][1].data.shape)


data, label = L.Data(source = test_db, backend = P.Data.LMDB, batch_size = 100, ntop = 2, mean_file = mean_proto)


if SVD_R > 0:
    # SVD
    print ("SVD %d" % SVD_R)
    u, sigma, vt = la.svd(net.params["ip2"][0].data)
    print ("Sigma: ", sigma)
    if SVD_R > len(sigma):
        print ("SVD_R is too large :-(")
        sys.exit()
    U = np.matrix(u[:, :SVD_R])
    S = np.matrix(np.diag(sigma[:SVD_R]))
    VT = np.matrix(vt[:SVD_R, :])
    print ("IP2", net.params["ip2"][0].data.shape) # 10, 64
    print ("U", U.shape)
    print ("S", S.shape)
    print ("VT", VT.shape)

    # y = Wx + b
    # y = U * S * VT * x + b
    # y = U * ((S * VT) * x) + b
    # y = U * (Z * x) + b
    Z = S * VT

    np.copyto(netSVD.params["ipZ"][0].data, Z)
    np.copyto(netSVD.params["ipU"][0].data, U)
    np.copyto(netSVD.params["ipU"][1].data, net.params["ip2"][1].data)

    nn = netSVD
else:
    print ("NORMAL")
    nn = net


n = len(testX)
pre = np.zeros(testy.shape)
print ("N = %d" % n)
for i in range(n):
    nn.blobs["data"].data[...] = testX[i] - mean_pic 
    nn.forward()
    prob = nn.blobs["prob"].data
    pre[i] = prob.argmax() 
    print ("%d / %d" % (i + 1, n))
right = np.sum(pre == testy) 
print ("Accuracy: %f" % (right * 1.0 / n))


if SVD_R > 0:
    np.save(RESULT_DIR + "net_SVD%d.npy" % SVD_R, pre)
else:
    np.save(RESULT_DIR + "net_normal.npy", pre)
