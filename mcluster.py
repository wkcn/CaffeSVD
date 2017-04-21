#coding=utf-8
import numpy as np
from sklearn.cluster import KMeans

class IP:
    def __init__(self, W, k, t = -np.inf):
        rows, cols = W.shape
        kmeans_model = KMeans(k)
        W_flat = W.reshape((rows * cols, 1))
        W_flat[W_flat < t] = 0.0
        kmeans_model.fit(W_flat)
        self.lb = kmeans_model.labels_
        self.centroids = np.zeros(k)
        for i in range(k):
            b = (self.lb == i)
            mean = np.mean(W_flat[b])
            self.centroids[i] = mean
        self.rows = rows
        self.cols = cols
        self.k = k
    def get_matrix(self):
        # 为了测试方便, 直接获取压缩后的矩阵
        mat = np.zeros(self.rows * self.cols)
        for i in range(self.k):
            mat[self.lb == i] = self.centroids[i]
        return mat.reshape(self.rows, self.cols)

def get_cluster_mat(W, k, t = -np.inf):
    rows, cols = W.shape
    kmeans_model = KMeans(k)
    W_flat = W.reshape((rows * cols, 1))
    W_flat[W_flat < t] = 0.0
    kmeans_model.fit(W_flat)
    lb = kmeans_model.labels_
    centroids = np.zeros(k)
    for i in range(k):
        b = (lb == i)
        mean = np.mean(W_flat[b])
        centroids[i] = mean

    mat = np.zeros(rows * cols)
    for i in range(k):
        mat[lb == i] = centroids[i]
    return mat.reshape(rows, cols)

def get_round_mat(W, k):
    return np.round(W, k)

if __name__ == "__main__":
    a = np.matrix("2.09,-0.98,1.48,0.09;0.05,-0.14,-1.08,2.12;-0.91,1.92,0,-1.03;1.87,0,1.53,1.49")
    '''
    ip = IP(a, 4) 
    print (ip.get_matrix())
    '''
    print get_cluster_mat(a, 4)
    print get_round_mat(a, 1)
