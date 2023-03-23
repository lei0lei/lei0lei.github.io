---
layout: default
title: import
permalink: /python/import
parent: python
has_toc: true
---
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>
模块化变成将一个大的编程任务分割成几个小的可管理的子任务和模块，单独的模块可以链接在一起组成大的应用。本文将介绍python模块化的两个机制:的module和package.

模块化变成有以下优点:
- 简单，将问题分割成更小的问题，减小问题领域，开发更容易更少出错。
- 可维护，不同的问题领域之间有逻辑边界，一个模块出问题不会影响程序的其他部分,更有利于多人合作。
- 可服用，一个模块定义的功能可以方便的被程序的其他部分进行复用，不需要多复制一份代码
- 作用域清晰，模块可以定义单独的命名空间，避免了标识符在程序的不同地方发生冲突。

在python中，函数，模块和包都是提升程序模块化的手段。

# Modules
python中模块的定义有三种方式:
- 用python写成
- 用C写然后在运行时进行动态加载，比如re模块
- 内建的模块直接包含在解释器中比如itertools模块

这三种方式都是通过`import`语句进行导入模块内容。本文主要介绍以python代码写成的模块，比如我们创建一个`mod.py`.

```py
s = "If Comrade Napoleon says it, it must be right."
a = [100, 200, 300]

def foo(arg):
    print(f'arg = {arg}')

class Foo:
    pass
```

```py
>>> import mod
>>> print(mod.s)
If Comrade Napoleon says it, it must be right.
>>> mod.a
[100, 200, 300]
>>> mod.foo(['quux', 'corge', 'grault'])
arg = ['quux', 'corge', 'grault']
>>> x = mod.Foo()
>>> x
<mod.Foo object at 0x03C181F0>
```

## 模块的搜索路径

当我们输入以下语句时会发生什么:

```py
import mod
```
解释器会从一个目录列表中搜索`mod.py`，目录列表如下:
- 如果使用解释器，当前目录或者输入脚本运行目录
- 在PYTHONPATH环境变量中包含的目录
- python安装时配置的依赖安装的列表目录

这个搜索路径可以从python的变量sys.path中查看。

```py
>>> import sys
>>> sys.path
['', 'C:\\Users\\john\\Documents\\Python\\doc', 'C:\\Python36\\Lib\\idlelib',
'C:\\Python36\\python36.zip', 'C:\\Python36\\DLLs', 'C:\\Python36\\lib',
'C:\\Python36', 'C:\\Python36\\lib\\site-packages']
```

为了保证调用时可以找到`mod.py`，需要做下面的其中一件事:
- 如果使用解释器，将`mod.py`放到输入脚本对应的目录或者当前目录
- 更改环境变量`PYTHONPATH`来包含`mod.py`所处的目录
- 将`mod.py`放到依赖安装的目录下

当然也可以在运行时更改`sys.path`.比如:

```py
>>> sys.path.append(r'C:\Users\john')
>>> sys.path
['', 'C:\\Users\\john\\Documents\\Python\\doc', 'C:\\Python36\\Lib\\idlelib',
'C:\\Python36\\python36.zip', 'C:\\Python36\\DLLs', 'C:\\Python36\\lib',
'C:\\Python36', 'C:\\Python36\\lib\\site-packages', 'C:\\Users\\john']
>>> import mod
```

导入一个模块之后，可以通过`__file__`属性查看文件从哪里导入的:

```py
>>> import mod
>>> mod.__file__
'C:\\Users\\john\\mod.py'

>>> import re
>>> re.__file__
'C:\\Python36\\lib\\re.py'
```
## import语句

## dir函数

## 将模块作为脚本执行

## 重新加载模块



# Packages



## 包初始化

## `import *`

## 子包

# Import

python的import如何工作的?当我们输入:

```py
import abc
```
- python首先在`sys.modules`查找名字`abc`，所有之前导入的模块名字会缓存在其中。
- 如果没在模块缓存中找到，将会在内建的模块(python标准库)中进行搜索
- 如果没找到会搜索sys.path定义的目录列表中搜索，这个列表一般包含当前目录，而且会首先搜索当前目录
- 在找到模块后会将其绑定到本地作用域上，没找到则会有`ModuleNotFoundError`


# Advanced import