#coding=utf-8
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import numpy as np
from numpy import linalg as la


CAFFE_HOME = "/opt/caffe/"

deploy = "./proto/cifar10_quick.prototxt"
deploySVD = "./proto/cifar10_SVD8.prototxt"
caffe_model = CAFFE_HOME + "/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5"

train_db = CAFFE_HOME + "examples/cifar10/cifar10_train_lmdb"
test_db = CAFFE_HOME + "examples/cifar10/cifar10_test_lmdb"
mean_proto = CAFFE_HOME + "examples/cifar10/mean.binaryproto"

# Load model and network
net = caffe.Net(deploy, caffe_model, caffe.TEST) 
netSVD = caffe.Net(deploySVD, caffe_model, caffe.TEST)

for layer_name, param in net.params.items():
    # 0 is weight, 1 is biase
    print (layer_name, param[0].data.shape)

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
print (net.params["ip2"][1].data)


data, label = L.Data(source = train_db, backend = P.Data.LMDB, batch_size = 100, ntop = 2, mean_file = mean_proto)
#print (netSVD.params["ipU"][0].data)

# SVD
print ("SVD")
print (net.params["ip2"][0].data.shape)
r = 8
u, sigma, vt = la.svd(net.params["ip2"][0].data)
U = u[:, :r]
S = np.diag(sigma[:r])
VT = vt[:r, :]

print(type(netSVD.params["ipU"][0].data))
print(type(netSVD.params["ipU"][0]))
print(dir(netSVD.params["ipU"][0]))
print(netSVD.params["ipU"][0].data.shape)
# y = Wx + b
# y = U * S * VT * x + b

np.copyto(netSVD.params["ipVT"][0].data, VT)
np.copyto(netSVD.params["ipS"][0].data, S)
np.copyto(netSVD.params["ipU"][0].data, U)
np.copyto(netSVD.params["ipU"][1].data, net.params["ip2"][1].data)


'''
# U Sigma VT
r = 8
rows, cols = net.params["ip2"][0].data.shape
# out: rows
print (rows, cols)
print (type(net.blobs["ip1"]))
U_layer = L.InnerProduct(net.blobs["ip1"], num_output = r)
S_layer = L.InnerProduct(U_layer, num_output = r) 
VT_layer = L.InnerProduct(S_layer, num_output = rows) 
loss_layer = L.SoftmaxWithLoss(VT_layer, label)
acc_layer = L.Accuracy(VT_layer, label)

print (dir(loss_layer))
print (loss_layer.to_proto)

print (dir(U_layer))
print (type(U_layer))
print (type(loss_layer))
print ("====") 
print (dir(loss_layer))

for layer_name, param in net.params.items():
    # 0 is weight, 1 is biase
    print (layer_name, param[0].data.shape)

# SVD
u, sigma, vt = la.svd(net.params["ip2"][0].data)
U = u[:, :r]
S = np.diag(sigma[:r])
VT = vt[:r, :]


# Set Weights:
U_layer[0].data = U
U_layer[1].data = np.zeros(U_layer[1].data.shape)
S_layer[0].data = S
S_layer[1].data = np.zeros(S_layer[1].data.shape)
VT_layer[0].data = VT
VT_layer[1].data = net.params["ip2"][1].data 
'''
