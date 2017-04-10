import os
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
