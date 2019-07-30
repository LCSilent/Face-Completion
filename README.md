##一.环境搭建
硬件：GTX1060、Core i5
软件：Windows10、TensorFlow1.4、Anaconda3.5、Python3.6、PyCharm、python_opencv、CUDA8.0、Cudnn6.0
环境配置：
+ 安装CUDA8.0和Cudnn6.0
+ 安装Python3.6版本的Anaconda3.5
+ 安装Tensorflow：在Anaconda中创建TensorFlow的虚拟环境，安装Tensorflow。
+ 将项目部署到Pycharm中即可。

##二.函数说明
+ faceproc 主要是对[celebA数据集](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)进行预处理，比如图片加上遮蔽部分作为输入。
+ G_network 主要是整个网络结构的模型，是对vgg的改进，加入反卷积层恢复图像的尺寸。
+ main主函数，分为训练和预测阶段。训练产生模型放入了model文件夹下，加载模型可以进行人脸修复预测。

##三.人脸修复项目说明
人脸修复，作为图像编辑的常见研究领域，主要是为了填补人脸中丢失或者被遮蔽的部分，使恢复的部分尽可能与原始人脸相近。主要使用基于卷积神经网络的人脸修复算法，这个模型主要使用编码-解码来构建的神经网络，通过输入一张随机遮蔽的人脸图像，修复得出一张复原的人脸图像。

+ 网络结构：网络主要由卷积层、池化层、激活层、全连接层、反卷积层组成的编码器和解码器结构，先通过编码结构提取图片的特征，在通过解码器恢复图片到原来的尺寸，其中上采样层是通过反卷积来实现的。
+ 网络的损失函数：本文用L1型损失函数和L2型损失函数做了一些对比，最终通过实验结果证明L1型损失函数的效果要好于L2型损失函数。
+ 网络训练过程：设置整个网络的batchsize大小为32，训练的时候每32张图片求平均的loss，来更新整个网络的权重。网络的初始学习率设置为0.0001，在网络的训练周期不断增加过程中，学习率会逐步下降，主要是为了防止产生局部震荡的线性。每30个周期会得到网络的训练模型，通过加载模型来测试测试集的图片。


## 四.实验结果
![avatar](https://note.youdao.com/yws/api/personal/file/WEBf5f3bcc4f147d33c85e89f7ab19c158e?method=getImage&version=9&cstk=ZS_MstR1)