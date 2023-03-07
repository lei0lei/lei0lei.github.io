---
layout: default
title: python style
permalink: /python/style
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
# 命名传统
## 变量、函数、包、类命名


## 下划线

- 单下划线

单下划线有时是来表明是一个临时变量或者占位变量。比如在下面的循环中不需要访问index，可以用`_`来替代。
```python
for _ in range(32):
... print('Hello, World.')
```
  也可以用在unpacking表达式中，表示要忽略这个位置解压出来的值，比如下面的代码,下划线就是一个占位变量:

```python
>>> car = ('red', 'auto', 12, 3812.4)
>>> color, _, _, mileage = car

>>> color
'red'
>>> mileage
3812.4
>>> _
12
```
除此之外单下划线还可以用作表示interpreter eval的最后一个表达式的结果。在解释器会话中可能用到。

```python
>>> 20 + 3
23
>>> _
23
>>> print(_)
23

>>> list()
[]
>>> _.append(1)
>>> _.append(2)
>>> _.append(3)
>>> _
[1, 2, 3]
```

- 前置单下划线
  
对于变量或者方法名，前置单下划线表只是一个习惯，完全不会影响程序的行为。前置单下划线告诉编程者`ok,这个变量或者方法只能在内使用`.这个习惯记录在`PEP 8`中。

python没有私有或者公有变量的区别，使用这个习惯就好像在说:`这不是一个类的公有接口，不要随便改动`。下面是一个例子:

```py
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 23
```
如果访问这两个属性会如何?随便访问，没有区别!

```py
>>> t = Test()
>>> t.foo
11
>>> t._bar
23
```
不过下划线影响了名字导入模块的方式:

```py
# This is my_module.py:

def external_func():
    return 23

def _internal_func():
    return 42
```
如果使用`import *`来引入模块中的所有名字,python不会import带前置下划线的名字，除非这个模块定义了`__all__` list来覆盖这个行为。

```py
>>> from my_module import *
>>> external_func()
23
>>> _internal_func()
NameError: "name '_internal_func' is not defined"
```
我们应该尽量避免'wildcard import'。正常的import不会被前置单下划线影响。

```py
>>> import my_module
>>> my_module.external_func()
23
>>> my_module._internal_func()
42
```

- 后置单下划线
  
有的时候一个变量的名字被关键字占用了，像`class`或者`def`无法作为变量名，这种情况下可以加一个单下划线避免冲突:

```py
>>> def make_object(name, class):
SyntaxError: "invalid syntax"

>>> def make_object(name, class_):
...     pass
```
这个规则在` PEP 8 `中有描述.

- 前置双下划线

这种命名方式不像前面只是一种传统，如果类的属性以双下划线开头，会有一些特殊含义。python解释器会重写属性名来避免在子类中出现名字冲突。

又叫做`name mangling`，解释器会改变变量的名使其很难在之后类扩展的过程中发生冲突。下面是一个例子:

```py
class Test:
    def __init__(self):
        self.foo = 11
        self._bar = 23
        self.__baz = 23
```
然后使用内建的`dir()`函数看一下对象的属性:

```py
>>> t = Test()
>>> dir(t)
['_Test__baz', '__class__', '__delattr__', '__dict__', '__dir__',
 '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
 '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__module__',
 '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__',
 '__setattr__', '__sizeof__', '__str__', '__subclasshook__',
 '__weakref__', '_bar', 'foo']
 ```

可以看到名字发生了改变，`__baz`发生了什么?

如果细看会发现变成了`_Test__baz`,这就是python使用了name mangling来保护父类不被子类更改变量。

首先创建一个类来扩展Test类并试图在构造器中覆盖现存的属性:

```py
class ExtendedTest(Test):
    def __init__(self):
        super().__init__()
        self.foo = 'overridden'
        self._bar = 'overridden'
        self.__baz = 'overridden'
```

现在，foo, _bar, 和 __baz 在扩展出来的类的实例是什么?

```py
>>> t2 = ExtendedTest()
>>> t2.foo
'overridden'
>>> t2._bar
'overridden'
>>> t2.__baz
AttributeError: "'ExtendedTest' object has no attribute '__baz'"
Wait, why did we get that AttributeError when we tried to inspect the value of t2.__baz? Name mangling strikes again! It turns out this object doesn’t even have a __baz attribute:

>>> dir(t2)
['_ExtendedTest__baz', '_Test__baz', '__class__', '__delattr__',
 '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__',
 '__getattribute__', '__gt__', '__hash__', '__init__', '__le__',
 '__lt__', '__module__', '__ne__', '__new__', '__reduce__',
 '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__',
 '__subclasshook__', '__weakref__', '_bar', 'foo', 'get_vars']
 ```
正如我们看到的`__baz`被变成了` got turned into `_ExtendedTest__baz`来防止意外的更改:

```py
>>> t2._ExtendedTest__baz
'overridden'
```

但是原始的 `_Test__baz`仍然存在:

```py
>>> t2._Test__baz
42
```

看下面的一个例子:

```py
class ManglingTest:
    def __init__(self):
        self.__mangled = 'hello'

    def get_mangled(self):
        return self.__mangled

>>> ManglingTest().get_mangled()
'hello'
>>> ManglingTest().__mangled
AttributeError: "'ManglingTest' object has no attribute '__mangled'"
```

name mangling会影响属性和方法名:

```py
class MangledMethod:
    def __method(self):
        return 42

    def call_it(self):
        return self.__method()

>>> MangledMethod().__method()
AttributeError: "'MangledMethod' object has no attribute '__method'"
>>> MangledMethod().call_it()
42
```

下面是另一个例子:

```py
_MangledGlobal__mangled = 23

class MangledGlobal:
    def test(self):
        return __mangled

>>> MangledGlobal().test()
23
```
首先创建了一个全局变量然后访问一个类的内部变量，由于name mangling就可以访问`__mangled`.python解释器会自动将`__mangled`扩展为`_MangledGlobal_mangled`.

{: .highlight }
>>⏰ Sidebar: What’s a “dunder” in Python?
>>I’ve you’ve heard some experienced Pythonistas talk about Python or watched a few conference talks you may have heard the term dunder. If you’re wondering what that is, here’s your answer:
>>
>>Double underscores are often referred to as “dunders” in the Python community. The reason is that double underscores appear quite often in Python code and to avoid fatiguing their jaw muscles Pythonistas often shorten “double underscore” to “dunder.”
>>
>>For example, you’d pronounce __baz as “dunder baz”. Likewise __init__ would be pronounced as “dunder init”, even though one might think it should be “dunder init dunder.” But that’s just yet another quirk in the naming convention.
>>
>>It’s like a secret handshake for Python developers 🙂

- 前后双下划线
如果使用了前后双下划线，命名修饰将不再适用，解释器不会对其有任何影响。

```python
class PrefixPostfixTest:
    def __init__(self):
        self.__bam__ = 42

>>> PrefixPostfixTest().__bam__
42
```

但是这种命名方式为保留方式，只能被语言特征使用，比如`__init__`是对象构造器，`__call__`是对象调用器。这种方法经常被叫做魔法方法(magic-method).
最好不要在自己的程序中使用这种方式命名，可能会与未来python的某个特性冲突。

# docstring