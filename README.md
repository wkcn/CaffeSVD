实现一个压缩的深度神经网络

- 动机
	深度神经网络中, 全连接网络层的参数占据的存储空间比较大, 在深度卷积神经网络中, 全连接层的参数占据了90%以上的参数.[1] 深度学习算法是计算密集型和存储密集型的[2], 使得它难以部署在拥有有限硬件资源的嵌入式系统上. 为了降低参数所占用的存储空间, 提高预测效率, 使用奇异值分解的方法(SVD)对全连接层的参数矩阵进行压缩, 并研究模型预测准确度与压缩率的关系.

- 实验环境
	操作系统: Arch Linux 4.10.8
	深度学习框架: Caffe(Apr 7,2017)
	编程语言: Python 2.7.13
	编译器: g++ (GCC) 6.3.1 20170306
			Cuda compilation tools, release 8.0, V8.0.44
	数据库: ldmb
	图像处理库: OpenCV3
	环境配置:
	CPU: Intel(r) Core(tm) i7-4702MQ CPU @ 2.20GHZ
	GPU: NVIDIA Corporation GK107M [GeForce GT 750M]

- 深度神经网络
-- 数据集
		CIFAR-10 dataset
-- 网络模型
		cifar10_quick_train_test.prototxt
![](net.png)

- 方法
	奇异值分解能够提取矩阵的重要特征, 它可以对任何形状的矩阵进行分解.
	使用了奇异解分解的方法对神经网络的全连接层进行压缩

- 改进

- 参考资料:
[1] Zichao Yang,Marcin Moczulski,Misha Denil,Nando de Freitas,Alex Smola,Le Song,Ziyu Wang. Deep Fried Convnets. Arxiv, 1412.7149, 2015
[2] Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. Arxiv, 1510.00149. 2016.
[3] LeftNotEasy, 机器学习中的数学(5)-强大的矩阵奇异值分解(SVD)及其应用, http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html
