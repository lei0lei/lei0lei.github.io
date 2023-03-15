---
layout: default
title: attrs
permalink: /MachineLearning/packages/attrs
parent: packages
grand_parent: python
has_toc: true
---
---
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>


attrs可以将我们从dunder的写法中解放出来.

# 概览
为了简化写类的过程，attrs提供了一个类装饰器以及在类上定义属性的声明式方式。

```py
>>> from attrs import asdict, define, make_class, Factory

>>> @define
    class SomeClass:
        a_number: int = 42
        list_of_numbers: list[int] = Factory(list)

        def hard_math(self, another_number):
            return self.a_number + sum(self.list_of_numbers) * another_number


>>> sc = SomeClass(1, [1, 2, 3])
>>> sc
SomeClass(a_number=1, list_of_numbers=[1, 2, 3])

>>> sc.hard_math(3)
19
>>> sc == SomeClass(1, [1, 2, 3])
True
>>> sc != SomeClass(2, [3, 2, 1])
True

>>> asdict(sc)
{'a_number': 1, 'list_of_numbers': [1, 2, 3]}

>>> SomeClass()
SomeClass(a_number=42, list_of_numbers=[])

>>> C = make_class("C", ["a", "b"])
>>> C("foo", "bar")
C(a='foo', b='bar')
```

在声明了属性之后`attrs`提供了:
- 简明的类属性概括
- 易读的`__repr__`
- 比较大小方法
- 初始化器
- ...

现在我们无需重复的写一些代码也没有运行时性能问题了，yeah.

不想要类型注解?没问题，对attrs类型是可选的，将`attrs.field`赋值给属性而不是使用类型标注。

上面的例子使用了attrs在20.1.0引入的api.相比与`dataclass`，attrs更加的灵活，比如可以为numpy数组定义比较运算符。

## 理念
- 创建更好的类,可以用于只有数据的容器比如`namedtuple`或者`types.SimpleNamespace`
- 只是一个附带了写好的方法的普通类
- 只跟踪dunder方法.
- 没有运行时影响

# 例子
最简单的使用方式是:

```py
>>> from attrs import define, field
>>> @define
    class Empty:
        pass
>>> Empty()
Empty()
>>> Empty() == Empty()
True
>>> Empty() is Empty()
False
```

让我们在类中加一些数据:

```py
>>> @define
    class Coordinates:
        x: int
        y: int
```

```py
>>> c1 = Coordinates(1, 2)
>>> c1
Coordinates(x=1, y=2)
>>> c2 = Coordinates(x=2, y=1)
>>> c2
Coordinates(x=2, y=1)
>>> c1 == c2
False
```

对于私有属性，attrs会加一个前置单下划线作为关键字参数，
```py
>>> @define
    class C:
        _x: int
>>> C(x=1)
C(_x=1)
```

如果想初始化自己的私有属性可以:
```py
>>> @define
    class C:
        _x: int = field(init=False, default=42)
>>> C()
C(_x=42)
>>> C(23)
Traceback (most recent call last):
   ...
TypeError: __init__() takes exactly 1 argument (2 given)
```