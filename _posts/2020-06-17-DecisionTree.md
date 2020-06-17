---
layout:     post
title:      DesicionTree
subtitle:   
date:       2020-06-17
author:     wang
header-img: img/post-bg-coffee.jpeg
catalog: true
tags: 
    - meachine learning
typora-root-url: ..
---

# DecisionTree（决策树）

## 什么是决策树

决策树（Decision Tree）是一种简单但是广泛使用的分类器。通过训练数据构建决策树，可以高效的对未知的数据进行分类和回归。

决策数有两大优点

* 决策树模型可以读性好，具有描述性，有助于人工分析
* 效率高，决策树只需要一次构建，反复使用，每一次预测的最大计算次数不超过决策树的深度。

如下图，朋友给你介绍了一个对象以及ta的长相，收入，身高情况，那么你怎么决定是否见和不见呢？

![img](/img/2020-06-17-DecisionTree/20170501132442447)

就上例而言，对一个对象，要判断三个条件：（1）长相（2）收入（3）身高

显然这三个条件（特征）的重要程度是不一样的，为什么首先判断长相这个条件呢?因为长相是你认为最重要的一个指标，收入其次，身高相对不重要。因此构建决策树的时候，越重要的特征越靠近根节点，选择特征的目标是尽量让分裂后的节点都属于同一类别。那么如何确定是重要的特征呢？






## 决策树常用算法

## 1. ID3（Iterative Dichotomiser 3）

该算法的核心是：**以信息增益为度量，选择分裂后信息增益最大的特征进行 分裂**

信息熵：

信息熵用来衡量信息量的大小
若不确定性越大，则信息量越大，熵越大
若不确定性越小，则信息量越小，熵越小
比如A班对B班，胜率一个为x，另一个为1-x
则信息熵为 -(xlogx + (1-x)log(1-x))
求导后容易证明x=1/2时取得最大，最大值为2
也就是说两者势均力敌时，不确定性最大，熵最大。

构建决策树的过程，就是减小信息熵，减小不确定性。从而完整构造决策树模型。更好的理解信息熵，参考链接[2]

![通俗理解信息熵- 知乎](data:image/png;base64,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)

信息增益



## 插入链接

[baidu](http://baidu.com)

## 粗体和斜体

Markdown 的粗体和斜体也非常简单，用两个 *包含一段文本就是粗体的语法，用一个 *包含一段文本就是斜体的语法。

**粗体**

*斜体*

## 表格

使用右键插入即可

| ID   | age  | dd   |
| ---- | ---- | ---- |
| 12   |      |      |
|      |      |      |
|      |      |      |

## 代码块

需要用两个 ` 把中间的代码包裹起来

```
def pp(str):
	print(str)
```

## 分割线

分割线的语法只需要另起一行，连续输入三个星号 *** 即可。

***

## 插入公式

参看链接 , chroma浏览器安装插件

[typora公式](https://blog.csdn.net/mingzhuo_126/article/details/82722455)

$sagadsg* x^2$





$$
x^2
y1
sag
$$

$$
\frac{1}{2}*\sin(\sum{a})
$$



dEND!

# 参考

[1]: https://blog.csdn.net/lisi1129/article/details/71055351	"决策树，随机森林简单原理"
[2]: https://charlesliuyx.github.io/2017/09/11/%E4%BB%80%E4%B9%88%E6%98%AF%E4%BF%A1%E6%81%AF%E7%86%B5%E3%80%81%E4%BA%A4%E5%8F%89%E7%86%B5%E5%92%8C%E7%9B%B8%E5%AF%B9%E7%86%B5/	"信息熵"
[3]: https://zhuanlan.zhihu.com/p/26596036	"信息增益"