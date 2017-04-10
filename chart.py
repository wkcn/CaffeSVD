from base import *
import matplotlib.pyplot as plt

label = np.load("label.npy")
n = len(label)
num_kinds = 10
# ip2 layer: 10 x 64
accs = []
maccs = []
mpres = []
mrecalls = []
mfs = []

# A(nxw) * B(wxm), time:O(nwm)  space:O(nw+wm)
# IP2(10 x 64) * x(64x1)     time:O(640) space:O(640) # exclude x
# U(10xr) * S(rxr) * VT(rx64) * x(64x1):
# time:O(r * 64 + r * r + 10 * r) = O(74 * r + r^2)
# space: O(10r + r^2 + 64r) = O(74r + r^2)


t = [i for i in range(3, 11)]
time_complex = [74 * r + r*r for r in t] 
print ("Time Complexity:")
print (time_complex)
space_complex = [74 * r + r*r for r in t] 
# 0 - 1
time_complex = np.array(time_complex) * 1.0 / max(time_complex)
space_complex = np.array(space_complex) * 1.0 / max(space_complex)
for r in t:
    name = "net_SVD%d.npy" % r
    pre  = np.load(name)
    acc, mean_acc, mean_precision, mean_recall, mean_F = eval_result(label, pre, num_kinds)
    accs.append(acc)
    maccs.append(mean_acc)
    mpres.append(mean_precision)
    mrecalls.append(mean_recall)
    mfs.append(mean_F)

plt.title("accuracy(totally), time and space complexity")
plt.plot(t, accs, "g", label = "accuracy")
plt.plot(t, time_complex, "r", label = "time complexity")
plt.plot(t, space_complex, "k", label = "space complexity")
plt.legend(loc="upper left")
plt.show()

plt.subplot(2,2,1)
plt.title("mean_accuracy")
plt.plot(t, maccs)

plt.subplot(2,2,2)
plt.title("mean_precision")
plt.plot(t, mpres)

plt.subplot(2,2,3)
plt.title("mean_recall")
plt.plot(t, mrecalls)

plt.subplot(2,2,4)
plt.title("mean_F")
plt.plot(t, mfs)

plt.show()
