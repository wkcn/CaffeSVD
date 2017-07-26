import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
z = np.load("ip1_SVD6_ipZ.npy")
rows, cols = z.shape
li = z.reshape(1,rows * cols).tolist()[0]
s = pd.Series(li)
print (s.describe())
plt.hist(li, bins = 100)
plt.show()
