import caffe
import sys
import numpy as np
if len(sys.argv) != 3:
    print ("convert_mean.py mean.binaryproto mean.npy")
    sys.exit()

blob = caffe.proto.caffe_pb2.BlobProto()
bin_mean = open(sys.argv[1], 'rb').read()
blob.ParseFromString(bin_mean)
arr = np.array(caffe.io.blobproto_to_array(blob))
npy_mean = arr[0]
print (npy_mean.shape)
np.save(sys.argv[2], npy_mean)
