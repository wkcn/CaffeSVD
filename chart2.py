from base import *
import matplotlib.pyplot as plt

label = np.load("label.npy")
n = len(label)
num_kinds = 10
# ip1 layer: 64 x 1024
accs = []
maccs = []
mpres = []
mrecalls = []
mfs = []

# A(nxw) * B(wxm), time:O(nwm)  space:O(nw+wm)
rows = 64
cols = 1024
SOURCE_SIZE = rows * cols


st = [i for i in range(1, 65)]
t = []
time_complex = []
space_complex = []
for r in range(1, min(rows, cols) + 1):
    name = "result/net_ip1_SVD%d.npy" % r
    if os.path.exists(name):
        pre  = np.load(name)
        acc, mean_acc, mean_precision, mean_recall, mean_F = eval_result(label, pre, num_kinds)
        accs.append(acc)
        maccs.append(mean_acc)
        mpres.append(mean_precision)
        mrecalls.append(mean_recall)
        mfs.append(mean_F)
        t.append(r)
        time_complex.append((rows + cols) * r)
        space_complex.append((rows + cols) * r)
# 0 - 1
time_complex_n = np.array(time_complex) * 1.0 / max(time_complex)
space_complex_n = np.array(space_complex) * 1.0 / max(space_complex)

acc, mean_acc, mean_precision, mean_recall, mean_F = eval_result(label, np.load("result/net_normal.npy"), num_kinds)
print ("r|\taccuracy|\tmean_precision|\tmean_recall|\tmean_F|\t|\ttime|\tspace|\tcompression_rate")
print ("---|---|---|---|---|---|---|---|---")
print ("source|\t%f|\t%f|\t%f|\t%f|\t%d|\t%d|\t%.2f%%" % (acc, mean_precision, mean_recall,mean_F, SOURCE_SIZE, SOURCE_SIZE, 100.0))
for i in range(len(t)-1,-1,-1):
    print ("%d|\t%f|\t%f|\t%f|\t%f|\t%d|\t%d|\t%.2f%%" % (t[i], accs[i], mpres[i], mrecalls[i],mfs[i], time_complex[i], space_complex[i], space_complex[i] * 100.0 / SOURCE_SIZE))

plt.title("accuracy(totally), time and space complexity")
plt.plot(t, accs, "g", label = "accuracy", linestyle="-")
plt.plot(t, time_complex_n, "r", label = "time complexity", linestyle="--")
plt.plot(t, space_complex_n, "k", label = "space complexity", linestyle="-.")
plt.legend(loc = "upper left")
plt.show()

plt.title("statistics")
plt.plot(t, accs, "r", label = "accuracy", linestyle="-")
#plt.plot(t, maccs, "r", label = "mean_accuracy")
plt.plot(t, mpres, "g", label = "mean_precision", linestyle="--")
plt.plot(t, mrecalls, "b", label = "mean_recall",linestyle="-.")
plt.plot(t, mfs, "y", label = "mean_F", linestyle=":")
plt.legend(loc = "upper left")

plt.show()
