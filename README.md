实现一个压缩的深度神经网络

# 动机
	深度神经网络中, 全连接网络层的参数占据的存储空间比较大, 在深度卷积神经网络中, 全连接层的参数占据了90%以上的参数.[1] 深度学习算法是计算密集型和存储密集型的[2], 使得它难以部署在拥有有限硬件资源的嵌入式系统上. 为了降低参数所占用的存储空间, 提高预测效率, 使用奇异值分解的方法(SVD)对全连接层的参数矩阵进行压缩, 并研究模型预测准确度与压缩率的关系.

# 实验环境
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

# 深度神经网络
- 数据集
		CIFAR-10 dataset
- 网络模型
	 * 训练网络时的神经网络定义
	cifar10_quick_train_test.prototxt
	conv1: data  -> conv1
	pool1: conv1 -> pool1
	relu1: pool1 -> pool1
	conv2: pool1 -> conv2
	relu2: conv2 -> conv2
	pool2: conv2 -> pool2
	conv3: pool2 -> conv3
	relu3: conv3 -> conv3
	pool3: conv3 -> pool3
	ip1  : pool3 -> ip1
	ip2  : ip1   -> ip2
	accuracy[TEST]: ip2, label -> accuracy
	loss(softmax with loss): ip2, label -> loss
	 * 训练结束后保存的神经网络的定义 
	conv1: data  -> conv1
	pool1: conv1 -> pool1
	relu1: pool1 -> pool1
	conv2: pool1 -> conv2
	relu2: conv2 -> conv2
	pool2: conv2 -> pool2
	conv3: pool2 -> conv3
	relu3: conv3 -> conv3
	pool3: conv3 -> pool3
	ip1  : pool3 -> ip1
	ip2  : ip1   -> ip2
	prob : ip2   -> prob


	



# 方法
- 奇异值(SVD)分解
奇异值分解能够提取矩阵的重要特征, 它可以对任何形状的矩阵进行分解.
$$A=U{\sum}V^T$$

- 部分奇异值分解
部分奇异值分解可以对矩阵的存储空间进行压缩
假设矩阵A的大小为mxn, 奇异值分解后得到:
$$A_{mxn}=U_{mxm}{\sum}_{mxn}{V^T}_{nxn}$$
$${\sum}$$为对角矩阵, 奇异值沿着对角线从大到小排列, 选取前r个奇异值来近似矩阵A, 有:
$$A_{mxn}{\approx}U_{mxr}{\sum}_{rxr}{V^T}_{rxn}$$
- 使用SVD对神经网络的全连接层进行压缩
设神经网络的某一层全连接层的参数矩阵为$$W \in R^{mxn}$$, 偏差向量为$$b \in R^m$$	
输入向量: $$x \in R^n$$, 输出向量: $$y \in R^m$$
全连接层输入输出的关系:
$$y = Wx + b$$
对W进行奇异值分解:
$$W = U_{mxm}{\sum}_{mxn}{V^T}_{nxn}$$
进行部分奇异值分解:
$$W_{mxn}{\approx}U_{mxr}{\sum}_{rxr}{V^T}_{rxn}$$
部分奇异值分解后, 原全连接层可以分为三个全连接层或两个全连接层, 但是不同的划分组合的时间复杂度与空间复杂度是不一样的


- 不同划分方式对应的时间复杂度与空间复杂度
划分方式							|层数| 时间复杂度       | 空间复杂度  
------------------------------------|-|------------------|-----------  
不压缩								|1|$$O(mn)$$ |$$O(mn)$$		 
$$(U_{mxr}{\sum}_{rxr}){V^T}_{rxn}$$|2|$$O(mr+rn)$$|$$O(mr+rn)$$
$$U_{mxr}({\sum}_{rxr}{V^T}_{rxn})$$|2|$$O(mr+rn)$$|$$O(mr+rn)$$
$$U_{mxr}{\sum}_{rxr}{V^T}_{rxn}$$|3|$$O(mr+r^2+rn)$$|$$O(mr+r^2+rn)$$

注: 上述表格中, 括号表示将多个矩阵合并为一个矩阵. 时间复杂度指的是一个输入经过原全连接层或压缩后生成的多个全连接层的时间复杂度, 由于参数矩阵比较大, 忽略偏差矩阵所占的空间以及计算时使用的时间.

从表格上可以看到, 当全连接层的参数矩阵被拆分为两层时, 其时间复杂度和空间复杂度比拆分成三层的情况低. 将一个矩阵拆分成两层全连接网络有两种方式, 它们的时间复杂度和空间复杂度是一样的, 当$$mr + rn < mn$$时, 即 $$r < \frac{mn}{m+n}$$时, 使用SVD压缩参数矩阵, 在时间复杂度和空间复杂度上有优势.


- 改进

- 参考资料:
[1] Zichao Yang,Marcin Moczulski,Misha Denil,Nando de Freitas,Alex Smola,Le Song,Ziyu Wang. Deep Fried Convnets. Arxiv, 1412.7149, 2015
[2] Song Han, Huizi Mao, William J. Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding. Arxiv, 1510.00149. 2016.
[3] LeftNotEasy, 机器学习中的数学(5)-强大的矩阵奇异值分解(SVD)及其应用, http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html
