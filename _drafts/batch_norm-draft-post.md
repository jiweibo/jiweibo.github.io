# Batch Normalization  
<br />
## 1. Why Batch Normalization
在训练深度神经网络的过程中，由于前一层参数的不断变化，使得网络每一层输入的分布不断发生变化，输入分布的变化使得参数需要不断调整，来补偿输入分布的变化，因此加剧了网络训练的难度。

我们将训练过程中由于参数的变化引起网络激活值分布变化这种行为称为<em>Internal Covariate Shift</em>。

Batch Normalization的目的就是减少<em>Internal Covariate Shift</em>

## 2. How Batch Normalization
对于网络的一层有d维的输入，我们要对每一维进行正则化，使得每一维均值为0，方差为1
$$公式1$$

注意如果简单的对每一层的输入进行正则化可能改变每一层的表征意义。例如，对sigmoid的输入进行正则化将会将它们限制在非线性曲线的线性区域。

引入变量gama beta对正则化后的值进行线性变换
$$公式2$$

直接引用论文中公式
$$算法1公式$$


反向传播公式
$$公式$$


## 3. Pytorch实现
Pytorch实现代码
对比是否使用Batch Normalization两次运行结果，我们发现使用batch norm后，收敛速度变快，并且最终结果有一定的提高。

