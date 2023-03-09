---
layout: default
title: OOP
permalink: /python/OOP
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


https://realpython.com/python3-object-oriented-programming/

https://realpython.com/python-super/


# 实例、类和静态方法

https://realpython.com/instance-class-and-static-methods-demystified/


# 重载

https://realpython.com/operator-function-overloading/

# 属性

https://realpython.com/python-property/

# 继承和多态

https://realpython.com/inheritance-composition-python/

# dataclass

https://realpython.com/python-data-classes/

# 构造器

https://realpython.com/python-class-constructor/

# 多重构造器

https://realpython.com/python-multiple-constructors/


# [metaclass](https://realpython.com/python-metaclasses/)

元编程是指程序知道自己本身或者有能力控制自身，python对类的元编程叫做metaclass.这是一种隐藏的OOP概念，在使用的时候可能根本意识不到或者说没有必要去意识到。

python提供了其他OOP语言不支持的能力:可以自定义metaclass.但是metaclass是比较有争议的:

```
“Metaclasses are deeper magic than 99% of users should ever worry about. If you wonder whether you need them, you don’t (the people who actually need them know with certainty that they need them, and don’t need an explanation about why).”
```

有的python学者相信我们永远不该使用自定义的metaclass,大多数情况下自定义的metaclass没有必要。

但是python的metaclass还是有必要了解，可能更好的了解python类的内部原理。

python中，类有两种变体，没有正式的名字，姑且叫做旧风格和新风格的类。

在旧风格的类中，类和类型不是一个相同的东西，旧风格的类的实例是用单独的内建类型`instance`实现的。如果`obj`是旧风格的类的实例，`obj.__class__`来指定类，但是`type(obj)`总是`instance`,如下面python2.7的例子:

```py
>>> class Foo:
...     pass
...
>>> x = Foo()
>>> x.__class__
<class __main__.Foo at 0x000000000535CC48>
>>> type(x)
<type 'instance'>
```
