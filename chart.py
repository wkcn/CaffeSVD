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
# IP2(10 x 64) * x(64x1)
# Time:O(640) Space:O(640) # exclude x
# W = U(10xr) * S(rxr) * VT(rx64)
# U(10xr) * Z(rx64) * x(64 x 1)
# Time: O(r*64 + 10 * r) = O(74r)
# Space: O(10r + 64r) = O(74r) 


t = [i for i in range(3, 11)]
time_complex = [74 * r for r in t] 
print ("Time Complexity:")
print (time_complex)
space_complex = [74 * r for r in t] 
# 0 - 1
time_complex_n = np.array(time_complex) * 1.0 / max(time_complex)
space_complex_n = np.array(space_complex) * 1.0 / max(space_complex)
for r in t:
    name = "result/net_SVD%d.npy" % r
    pre  = np.load(name)
    acc, mean_acc, mean_precision, mean_recall, mean_F = eval_result(label, pre, num_kinds)
    accs.append(acc)
    maccs.append(mean_acc)
    mpres.append(mean_precision)
    mrecalls.append(mean_recall)
    mfs.append(mean_F)

print ("k\taccuracy\tmean_precision\tmean_recall\tmean_F\ttime complexity\tspace complexity")
for i in range(len(t)-1,-1,-1):
    print ("%d\t%f\t%f\t%f\t%f\t%d\t%d" % (t[i], accs[i], mpres[i], mrecalls[i],mfs[i], time_complex[i], space_complex[i]))

plt.title("accuracy(totally), time and space complexity")
plt.plot(t, accs, "g", label = "accuracy")
plt.plot(t, time_complex_n, "r", label = "time complexity")
plt.plot(t, space_complex_n, "k", label = "space complexity")
plt.legend(loc = "upper left")
plt.show()

plt.title("statistics")
plt.plot(t, accs, "r", label = "accuracy")
#plt.plot(t, maccs, "r", label = "mean_accuracy")
plt.plot(t, mpres, "g", label = "mean_precision")
plt.plot(t, mrecalls, "b", label = "mean_recall")
plt.plot(t, mfs, "y", label = "mean_F")
plt.legend(loc = "upper left")

plt.show()
