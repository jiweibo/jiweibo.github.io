---
layout: post
title:  "DecisionTree"
date:   2018-08-27 10:00:00 +0800
description: 决策树
categories: Machine Learning
tags: [Machine Learning, DecisionTree]
location: Harbin,China
img: ml.jpeg
---

# 决策树

决策树学习算法包含特征选择,决策树的生成和决策树的剪枝过程.

决策树学习常用的算法有ID3,C4.5和CART,下面一一介绍.

## 特征选择

### 熵

为了便于说明,先给出熵与条件熵的定义.

熵是表示随机变量不确定性的度量. 设X是一个取有限个值的离散随机变量,其概率分布为

$$P(X=x_i)=p_i,i=1,2,...,n$$

则随机变量X的熵定义为

$$H(X)=-\sum_{i=1}^np_ilogp_i$$

条件熵$H(Y\|X)$表示在已知随机变量X的条件下随机变量Y的不确定性.

$$H(Y|X)=\sum_{i=1}^np_iH(Y|X=x_i)$$

其中$p_i=P(X=x_i),i=1,2...,n$

### 信息增益

信息增益定义为集合D的经验熵$H(D)$与特征A给定的条件下D的经验条件熵$H(D\|A)$之差

$$g(D,A)=H(D)-H(D|A)$$

设训练数据集为D,$\|D\|$表示样本容量.

设有$K$个类$C_k,k=1,2...,K$,$C_k$为属于类$C_k$的样本个数.

设特征A有n个不同取值$\{a_1,a_2,...,a_n\}$,根据特征A的取值将D分为n个子集$D_1,D_2,...,D_n$, $\|D_i\|$ 为$D_i$的样本个数.

记子集$D_i$中属于类$C_k$的样本的集合为$D_{ik}$,$\|D_{ik}\|$为$D_{ik}$的样本个数.

信息增益算法如下:

1. 计算数据集D的经验熵 $H(D)$

$$H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}log_2\frac{|C_k|}{|D|}$$

2. 计算特征A对数据集D的经验条件熵$H(D\|A)$

$$H(D|A)=\sum_{i=1}^n\frac{|D_i|}{|D|}H(D_i)=-\sum_{i=1}^n\frac{|D_i|}{|D|}\sum_{k=1}^K\frac{|D_{ik}|}{|D_i|}log_2\frac{|D_{ik}|}{|D_i|}$$

3. 计算信息增益

$$g(D,A)=H(D)-H(D|A)$$

### 信息增益比

特征A对训练数据集D的信息增益比$g_r(D,A)$定义为其信息增益$g(D,A)$与其训练数据集D的经验熵$H(D)$之比

$$g_r(D,A)=\frac{g(D,A)}{H(D)}$$


## 决策树生成

### ID3算法

1. 若训练集D中所有实例属于同一类$C_k$,则决策树T为单节点树,并将类$C_k$作为该节点的类标记,返回T;
2. 若特征$A=\emptyset$,则T为单节点树,并将D中实例数最大的类$C_k$作为该节点的类标记,返回T;
3. 否则,计算A中各特征对D的信息增益,选择信息增益最大的特征$A_g$;
4. 如果$A_g$的信息增益小于阈值$\varepsilon$,则置T为单节点树,并将D中实例数最大的类$C_k$作为该节点的类标记,返回T;
5. 否则,对$A_g$的每一可能值$a_i$,依$A_g=a_i$将D分割为若干非空子集$D_i$,将$D_i$中实例数最大的类作为标记,构建子结点,由结点及其子结点构成树T,返回T;
6. 对第i个子结点,以$D_i$为训练集,以$A-\{A_g\}$为特征集,递归调用1-5,得到子树$T_i$,返回$T_i$

### C4.5算法

C4.5算法对ID3算法进行了改进,在生成过程中,用信息增益比来选择特征

1. 若训练集D中所有实例属于同一类$C_k$,则决策树T为单节点树,并将类$C_k$作为该节点的类标记,返回T;
2. 若特征$A=\emptyset$,则T为单节点树,并将D中实例数最大的类$C_k$作为该节点的类标记,返回T;
3. 否则,计算A中各特征对D的信息增益比,选择信息增益比最大的特征$A_g$;
4. 如果$A_g$的信息增益小于阈值$\varepsilon$,则置T为单节点树,并将D中实例数最大的类$C_k$作为该节点的类标记,返回T;
5. 否则,对$A_g$的每一可能值$a_i$,依$A_g=a_i$将D分割为若干非空子集$D_i$,将$D_i$中实例数最大的类作为标记,构建子结点,由结点及其子结点构成树T,返回T;
6. 对第i个子结点,以$D_i$为训练集,以$A-\{A_g\}$为特征集,递归调用1-5,得到子树$T_i$,返回$T_i$

C4.5算法较ID3算法改进总结

* 用信息增益率来选择属性，克服了用信息增益选择属性偏向选择多值属性的不足
* 在构造树的过程中进行剪枝
* 对连续属性进行离散化
* 能够对不完整的数据进行处理

## 决策树剪枝

决策树生成算法递归产生的决策树经常出现过拟合现象,需要对生成的进行简化操作,即剪枝.

设树T的叶节点个数为$\|T\|$,t是树T的叶节点,该叶结点有$N_t$个样本,其中k类的样本有$N_{tk}$个,k=1,2,...,K,$H_t(T)$为叶节点t上的经验熵, $\alpha\ge0$为参数

决策树学习的损失函数定义

$$C_{\alpha}(T)=\sum_{t=1}^{|T|}N_tH_t(T)+\alpha|T|$$

其中

$$H_t(T)=-\sum_k\frac{N_{tk}}{N_t}log\frac{N_{tk}}{N_t}$$

$C_{\alpha}(T)$第一项表示模型对训练数据的拟合程度,第二项表示模型复杂度

### 树的剪枝算法

1. 计算每个结点的经验熵
2. 递归的从树的叶节点向上回缩

   设一组叶节点回缩到其父结点之后,损失函数不变或变小,则进行剪枝,即将父结点变为新的叶结点

3. 返回2直至不能继续,得到损失函数最小的子树$T_{\alpha}$

## CART算法

CART模型既可以用于分类,也可以用于回归.

CART假设决策树是二叉树,内部结点的特征取值为"是"和"否",左分支取"是"分支,右分支取"否"分支.决策树等价于递归地二分每个特征,将输入空间即特征空间划分为有限个单元,并在这些单元上确定预测的概率分布,也就是在输入给定的条件下输出的条件概率分布.

CART算法由两步组成

1. 决策树生成
2. 决策树剪枝

### CART生成

#### 回归树的生成

一个回归树对应着输入空间的一个划分以及在划分单元上的输出值.假设已将输入空间划分为M个单元$R_1,R_2,...,R_M$,并且在每个单元$R_M$上有一个固定的输出值$c_m$,于是回归树模型可表示为

$$f(x)=\sum_{m=1}^Mc_mI(x\in R_m)$$

当输入空间的划分确定时,可以用平方误差$\sum_{x_i\in R_m}^M(y_i-f(x_i))^2$来表示回归树对于训练数据的预测误差,用平方误差最小的准则求解每个单元上的最优输出值.

单元$R_m$上的$c_m$的最优值$\hat{c}_m$是$R_m$上所有输入实例$x_i$对应的输出$y_i$的均值,即

$$\hat{c}_m=ave(y_i|x_i\in R_m)$$

那么,如何对输入空间进行划分那?

选择第j个变量$x^{(j)}$和它取得值s,作为切分变量和切分点,并定义两个区域

$R_1(j,s)=\{x\|x^{(j)}\le s\}$和$R_2(j,s)=\{x\|x^{(j)}\gt s\}$

然后寻找最优切分变量j和最优切分点s.具体地,求解

$$\min\limits_{j,s}[\min\limits_{c_1}\sum_{x_i\in R_1(j,s)}(y_i-c_1)^2+\min\limits_{c_2}\sum_{x_i\in R_2(j,s)}(y_i-c_2)^2]$$

对固定得输入j可以找到最优切分点s,$c_1$和$c_2$

遍历所有输入变量,找到最优得切分变量j,构成一个对(j,s).依次将输入空间划分为两个区域.接着,对每个区域重复上述划分过程,直到满足停止条件为止.

最小二乘回归树生成算法:

1. 选择最优切分变量j和切分点s,求解

$$\min\limits_{j,s}[\min\limits_{c_1}\sum_{x_i\in R_1(j,s)}(y_i-c_1)^2+\min\limits_{c_2}\sum_{x_i\in R_2(j,s)}(y_i-c_2)^2]$$

遍历变量j,对固定的切分变量j和切分点s,选择使上式达到最小值的对(j,s)

2. 用选定的对(j,s)划分区域并决定相应的输出值:
   
   $$R_1(j,s)=\{x|x^{(j)}\le s\}, R_2(j,s)=\{x|x^{(j)}\gt s\}$$

   $$\hat{c}_m=\frac{1}{N_m}\sum_{x_i\in R_m(j,s)}y_i,x\in R_m, m=1,2$$

3. 继续对两个子区域调用步骤1和2,直至满足停止条件.
4. 将输入空间划分为M个区域$R_1,R_2,...,R_M$,生成决策树:
   
   $$f(x)=\sum_{m=1}^M\hat{c}_mI(x\in R_m)$$


#### 分类树的生成

分类树用基尼指数选择最优特征,同时决定该特征的最优二值切分点

分类问题中假设有K个类,样本点属于第k类的概率为$p_k$,则概率分布的基尼指数定义为:

$$Gini(p)=\sum_{k=1}^Kp_k(1-p_k)=1-\sum_{k=1}^Kp_k^2$$

对于给定的样本集合D,其基尼指数为:

$$Gini(D)=1-\sum_{k=1}^K(\frac{|C_k|}{|D|})^2$$

$C_k$是D中属于第k类的样本子集,K是类的个数

如果样本集合D根据特征A是否取某一可能值a被分割为$D_1$和$D_2$两部分,即

$$D_1=\{(x,y)\in D| A(x)=a\}, D_2=D-D_1$$

则在特征A的条件下,集合D的基尼指数定义为

$$Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)$$

基尼指数Gini(D)表示集合D的不确定性,基尼指数越大,样本集合不确定性越大,这一点与熵相似

CART生成算法:
1. 设结点的训练数据集为D,计算现有特征对该数据集的基尼指数.此时,对每一个特征A,对其可能取的每个值a,根据样本点对A=a的测试为"是"或"否"将D分割为$D_1$和$D_2$两部分,根据下式计算A=a时的基尼指数

$$Gini(D,A)=\frac{|D_1|}{|D|}Gini(D_1)+\frac{|D_2|}{|D|}Gini(D_2)$$

2. 在所有可能的特征A以及它们所有可能的切分点a中,选择基尼指数最小的特征及其对应的切分点作为最优特征和最优切分点.依最优特征和最优切分点,从现结点生成两个子结点,将训练数据集依特征分配到两个子结点中
3. 对两个子结点递归调用1和2,直至满足停止条件
4. 生成CART决策树

算法停止条件

1. 结点中的样本数小于预定阈值
2. 样本集的基尼指数小于预定阈值
3. 没有更多特征


## CART剪枝

CART剪枝算法由两步组成:首先从生成算法产生的决策树$T_0$底端开始不断剪枝,直到$T_0$的根节点,形成一个子树序列$\{T_0,T_1,...,T_n\}$;然后通过交叉验证在独立的验证集上对子树序列进行测试,选择最优子树.
1. 剪枝,形成一个子树序列

在剪枝过程中,计算子树的损失函数:

$$C_{\alpha}(T)=C(T)+\alpha|T|$$

C(T)为对训练数据的预测误差,$C_{\alpha}(T)$为参数是$\alpha$时的子树T的整体损失.参数$\alpha$权衡训练数据的拟合程度与模型的复杂度

对固定的$\alpha$一定存在使损失函数$C_{\alpha}(T)$最小的子树,将其表示为$T_{\alpha}$.当$\alpha$大的时候,最优子树$T_{\alpha}$偏小;当$\alpha$小的时候,最优子树$T_{\alpha}$偏大.

可以用递归的方法对树进行剪枝.将$\alpha$从小增大,$0=\alpha_0\lt \alpha_1\lt ... \lt \alpha_n \lt +\infty$,产生一系列的区间$[\alpha_i,\alpha_{i+1}),i=0,1...,n$的最优子树序列$\{T_0,T_1,...,T_n\}$,序列中的子树是嵌套的.

具体的,从整体树$T_0$开始剪枝.对$T_0$的任意内部结点t,以t为单位结点树的损失函数为:

$$C_{\alpha}(t)=C(t)+\alpha$$

以t为根节点的子树$T_t$的损失函数为:

$$C_{\alpha}(T_t)=C(T_t)+\alpha|T_t|$$

当$\alpha=0$及$\alpha$充分小时,有不等式

$$C_{\alpha}(T_t)<C_{\alpha}(t)$$

当$\alpha$增大时,在某一$\alpha$有

$$C_{\alpha}(T_t)=C_{\alpha}(t)$$

当$\alpha$再增大时,不等式反向.只要$\alpha=\frac{C(t)-C(T_t)}{\|T_t\|-1}$,$T_t$与t有相同的损失函数值,而t得结点少,因此对$T_t$进行剪枝.

为此,对$T_0$中每一内部结点t,计算

$$g(t)=\frac{C(t)-C(T_t)}{|T_t|-1}$$

它表示剪枝后整体损失函数减少得程度.在$T_0$中减去g(t)最小得$T_t$,将得到的子树作为$T_1$,同时将最小的g(t)设为$\alpha_1$.$T_1$为区间$[\alpha_1,\alpha_2)$的最优子树

如此剪枝下去,直至得到根结点.

2. 在剪枝得到的子树序列$\{T_0,T_1,...,T_n\}$中通过交叉验证选取最优子树$T_{\alpha}$


综上,写出CART剪枝算法

1. 设k=0, $T=T_0$
2. 设$\alpha=+\infty$
3. 自上而下地对各内部结点t计算$C(T_t),\|T_t\|$以及

$$g(t)=\frac{C(t)-C(T_t)}{|T_t|-1}$$

$$\alpha=min(\alpha, g(t))$$

4. 自上而下地访问内部结点t,如果$g(t)=\alpha$,进行剪枝,并对叶结点t以多数表决法决定其类,得到树T.
5. 设k=k+1, $\alpha_k=\alpha,T_k=T$
6. 如果T不是由根节点单独构成的树,返回4
7. 采用交叉验证法在子树序列$\{T_0,T_1,...,T_n\}$中选取最优子树$T_{\alpha}$


## Code

[https://github.com/jiweibo/MachineLearningStudy/tree/master/decision%20tree](https://github.com/jiweibo/MachineLearningStudy/tree/master/decision%20tree)

## 引用和致谢

[1] 李航. 统计学习方法

[2] [https://github.com/apachecn/AiLearning](https://github.com/apachecn/AiLearning)

[3] [https://geekcircle.org/machine-learning-interview-qa/questions/14.html](https://geekcircle.org/machine-learning-interview-qa/questions/14.html)