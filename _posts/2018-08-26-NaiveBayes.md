---
layout: post
title:  "NaiveBayes"
date:   2018-08-26 21:00:00 +0800
categories: Machine Learning
location: Harbin,China
description: NaiveBayes 
---

# 朴素贝叶斯

朴素贝叶斯是基于贝叶斯定理和特征条件独立性假设的分类方法。

给定训练数据集,基于特征独立性假设学习输入输出联合概率分布;然后基于此模型,对给定的输入$x$,利用贝叶斯定理求出后验概率最大的$y$.

## 基本方法

条件概率分布的朴素贝叶斯的独立性假设

$$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)}|Y=c_k)=\prod_{i=1}^nP(X^{(i)}=x^{(i)}|Y=c_k)$$

先验概率分布

$$P(Y=c_k),k=1,2,...,K$$

后验概率

$$P(Y=c_k|X=x)=\frac{P(X=x|Y=c_k)P(Y=c_k)}{\sum_kP(X=x|Y=c_k)P(Y=c_k)}$$

后验概率最大化即期望风险最小化

$$y=arg\max\limits_{c_k}P(Y=c_k|X=x)$$

## 朴素贝叶斯法的参数估计

### 极大似然估计

先验概率的极大似然估计

$$P(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)}{N},k=1,2,...,K$$

条件概率的极大似然估计

$$P(X^{(j)}=a_{jl}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)}{\sum_{i=1}^NI(y_i=c_k)},j=1,2...,n;l=1,2,...,S_j;k=1,2,...K$$

### 学习与分类算法

1. 计算先验概率和条件概率
2. 对于给定的实例，计算后验概率
3. 后验概率最大的类作为输出

### 贝叶斯估计

极大似然估计可能会出现所要估计的概率值为０的情况,故常用贝叶斯估计,$\lambda$常取１

条件概率的贝叶斯估计

$$P_\lambda(X^{(j)}|Y=c_k)=\frac{\sum_{i=1}^NI(x_i^{(j)}=a_{jl},y_i=c_k)+\lambda}{\sum_{i=1}^NI(y_i=c_k)+S_j\lambda},j=1,2...,n;l=1,2,...,S_j;k=1,2,...K$$

先验概率的贝叶斯估计

$$P(Y=c_k)=\frac{\sum_{i=1}^NI(y_i=c_k)+\lambda}{N+K\lambda},k=1,2,...,K$$

## Code
[https://github.com/jiweibo/MachineLearningStudy/tree/master/naive%20bayes](https://github.com/jiweibo/MachineLearningStudy/tree/master/naive%20bayes)

## 引用和致谢

[1] 李航. 统计学习方法

[2] [https://github.com/apachecn/AiLearning](https://github.com/apachecn/AiLearning)