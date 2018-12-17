---
layout: post
title:  "Effective C++————习惯C++"
date:   2018-12-17 15:22:00 +0800
description: Effective C++
categories: Effective C++
tags: [C++, Effective C++]
location: Harbin,China
img: c++.png
---

# 条款01：视C++为一个语言联邦

将C++视为一个由相关语言组成的联邦而非单一语言。

* C
* Object-Oriented C++
* Template C++
* STL


# 条款02：尽量以 const,enum,inline 替换 #define

* #define所使用的名称可能并未进入symbol table，调试增加困难
* #define不重视作用域，不能提供封装性，无法利用#define创建一个class的专属常量

{% highlight C++ %}
// static class常量 完成 in class 初值设定
//hpp
class GamePlayer{
    static const int NumTurns = 5; //常量声明
    int scores[NumTurns];
    ...
};
//cpp
const int GamePlayer::NumTurns; //NumTurns定义
{% endhighlight %}

{% highlight C++ %}
//hpp
class CostEstimate{
private:
    static const double FudgeFactor; // staic class常量声明，位于头文件内
    ...
};

//cpp
const double CostEstimate::FudgeFactor = 1.35;  // 位于实现文件内
{% endhighlight %}

{% highlight C++ %}
// enum hack
class GamePlayer{
    enum {NumTurns=5};  // enum hack方法令NumTurns成为5的记号
    int scores[NumTurns];
    ...
};
{% endhighlight %}

* #define实现宏很容易出错，建议用template<> inline代替


# 条款03：尽可能使用const

{% highlight C++ %}
char greeting[] = "Hello";
char* p = greeting; // non-const pointer, non-const data
const char* p = greeting;   //non-const pointer, const data
char* const p = greeting;   // const pointer,non-const data
const char* const p = greeting; // const pointer, const data
{% endhighlight %}

### const 成员函数

bitwise const 以及 logical const，这部分较为复杂，查看书籍深入理解。

### 在const 和 non-const成员函数中避免重复

当const和non-const成员函数有着实质等价的实现时，令non-const版本调用const版本可避免代码重复


# 条款04：确定对象被使用前已先被初始化

* 为内置数据对象进行手工初始化

* 构造函数中最好使用成员初始化列进行初始化，而不要在构造函数本体内使用赋值操作。初始化列中的成员变量，其排列次序应该和它们在class中的声明次序相同

{% highlight C++ %}
class PhoneNumber {...};
class ABEntry{
public:
    ABEntry(const std::string& name, const std::string& address,
    const std::list<PhoneNumber>& phones);
private:
    std::string theName;
    std::string theAddress;
    std::list<PhoneNumber> thePhones;
    int numTimesConsulted;
};

//ABEntry::ABEntry(const std::string& name, const std::string& address,
//    const std::list<PhoneNumber>& phones)
//{
//    theName = name;         //这些都是赋值assignments而非初始化initializations
//    theAddress = address;
//    thePhones = phones;
//    numTimesConsulted = 0;
//}

ABEntry::ABEntry(const std::string& name, const std::string& address,
    const std::list<PhoneNumber>& phones)
    :theName(name),         // initializations
    theAddress(address),
    thePhones(phones),
    numTimesConsulted(0)
{

}
{% endhighlight %}

* 为免除'跨编译单元初始化次序'问题，以local static对象代替non-local static对象

假设有一单一文件系统类
{% highlight C++ %}
class FileSystem{       // 来自程序库
public:
    ...
    std::size_t numDisks() const;
    ...
};
extern FileSystem tfs;  //预备给客户使用的对象
{% endhighlight %}

假设客户建立了一个class以处理文件系统内的目录，很自然他们的class会用上tfs对象
{% highlight C++ %}
class Director{         //由程序库客户建立
public:
    Directory(params);
    ...
};
Directory::Directory(params)
{
    ...
    std::size_t disks = tfs.numDisks();     //使用tfs对象
    ...
}
{% endhighlight %}


客户决定创建一个Directory对象，用来放置临时文件
```
Directory tempDir(params);      //为临时文件
```
除非tfs在tempDir前被初始化，否则tempDir就会用到尚未初始化的tfs。但是tfs和tempDir由不同的人在不同的时间于不同的源码文件内建立起来，如何确保tfs会在tempDir前先被初始化呐？

可以将每个non-local static对象搬到自己专属函数内，这些函数返回一个reference指向它所含的对象，换句话说non-local static对象被local static对象替换了，这也是**Singleton**模式的常见手法

{% highlight C++ %}
class FileSystem {...}; //同前
FileSystem& tfs()       //这个函数替换tfs对象
{
    static FileSystem fs;   //local static 对象
    return fs;              //返回一个reference指向上述对象
}
class Directory {...};  //同前
Directory::Directory(params)
{
    ...
    std::size_t disks = tfs().numDisks();
    ...
}
Directory& tempDir()
{
    static Directory td;
    return td;
}
{% endhighlight %}


这种方法很好的解决static对象初始化顺序问题

但是因为内含static对象，使得在多线程系统中带有不确定性，再说一次，任何一种non-const static对象，不论是local还是non-local，在多线程环境下等待某事发生都会出问题


# 引用

Effective C++改善程序与设计的55个具体做法（第三版）