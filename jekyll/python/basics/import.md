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
### `import <module_name>`
最简单的import语句如下:
```py
import <module_name>
```
这句话还无法让模块内容被调用者直接可以访问，每个模块都有自己的私有符号表，作为模块中定义的所有对象的全局符号表，因此导入的是一个独立的命名空间。
`import <module_name>`只是把模块名放到调用者的符号表中，定义在模块中的对象还是在模块的私有符号表中。

从调用者的角度看模块中的对象必须使用`.`符号来访问额模块的内容。

```py
>>> import mod
>>> mod
<module 'mod' from 'C:\\Users\\john\\Documents\\Python\\doc\\mod.py'>
>>> s
NameError: name 's' is not defined
>>> foo('quux')
NameError: name 'foo' is not defined
```
这句话会把mod放到本地符号表中。
但是`s`和`foo`仍然在模块的私有符号表中，而不在本地的上下文中。

要在本地上下文中访问必须使用以下的形式:
```py
>>> mod.s
'If Comrade Napoleon says it, it must be right.'
>>> mod.foo('quux')
arg = quux
```
可以使用逗号分割导入多个模块，`import <module_name>[, <module_name> ...]`.

### `from <module_name> import <name(s)>`
要直接把模块中的名字导入调用者的符号表中可以使用:
```py
from <module_name> import <name(s)>
```
```py
>>> from mod import s, foo
>>> s
'If Comrade Napoleon says it, it must be right.'
>>> foo('quux')
arg = quux

>>> from mod import Foo
>>> x = Foo()
>>> x
<mod.Foo object at 0x02E3AD50>
```
这很容易符号已经存在在本地符号表中的名字，比如:
```py
>>> a = ['foo', 'bar', 'baz']
>>> a
['foo', 'bar', 'baz']

>>> from mod import a
>>> a
[100, 200, 300]
```
甚至还有一种更讨厌的写法:
```py
from <module_name> import *
```
很容易忽略掉被覆盖的名字。
```py
>>> from mod import *
>>> s
'If Comrade Napoleon says it, it must be right.'
>>> a
[100, 200, 300]
>>> foo
<function foo at 0x03B449C0>
>>> Foo
<class 'mod.Foo'>
```
### `from <module_name> import <name> as <alt_name>`
最推荐的写法如下:
```py
from <module_name> import <name> as <alt_name>[, <name> as <alt_name> …]
```
```py
>>> s = 'foo'
>>> a = ['foo', 'bar', 'baz']

>>> from mod import s as string, a as alist
>>> s
'foo'
>>> string
'If Comrade Napoleon says it, it must be right.'
>>> a
['foo', 'bar', 'baz']
>>> alist
[100, 200, 300]
```
### `import <module_name> as <alt_name>`

也可以单独再起一个别名:
```py
>>> import mod as my_module
>>> my_module.a
[100, 200, 300]
>>> my_module.foo('qux')
arg = qux
```
也可以在函数调用的时候再进行`import`,这样包只在函数调用的时候才会可见。

```py
>>> def bar():
...     from mod import foo
...     foo('corge')
...

>>> bar()
arg = corge
```
但是在python3中不支持如下写法:
```py
>>> def bar():
...     from mod import *
...
SyntaxError: import * only allowed at module level
```
如果希望避免异常的导入可以使用:
```py
>>> try:
...     # Non-existent module
...     import baz
... except ImportError:
...     print('Module not found')
...

Module not found
```

```py
>>> try:
...     # Existing module, but non-existent object
...     from mod import baz
... except ImportError:
...     print('Object not found in module')
...

Object not found in module
```

## dir函数
内建的`dir`函数可以返回命名空间中的名字列表，不带参数的话返回的是本地符号表:
```py
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']

>>> qux = [1, 2, 3, 4, 5]
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'qux']

>>> class Bar():
...     pass
...
>>> x = Bar()
>>> dir()
['Bar', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'qux', 'x']
```

这个函数可以用来确定import到底导入了那些名字:
```py
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']

>>> import mod
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'mod']
>>> mod.s
'If Comrade Napoleon says it, it must be right.'
>>> mod.foo([1, 2, 3])
arg = [1, 2, 3]

>>> from mod import a, Foo
>>> dir()
['Foo', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'a', 'mod']
>>> a
[100, 200, 300]
>>> x = Foo()
>>> x
<mod.Foo object at 0x002EAD50>

>>> from mod import s as string
>>> dir()
['Foo', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'a', 'mod', 'string', 'x']
>>> string
'If Comrade Napoleon says it, it must be right.'
```
列出模块中的名字:
```py
>>> import mod
>>> dir(mod)
['Foo', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
'__name__', '__package__', '__spec__', 'a', 'foo', 's']
```

```py
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']
>>> from mod import *
>>> dir()
['Foo', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'a', 'foo', 's']
```

## 将模块作为脚本执行
包含模块的py文件又叫做脚本，因此也是可以执行的。`mod.py`:
```py
s = "If Comrade Napoleon says it, it must be right."
a = [100, 200, 300]

def foo(arg):
    print(f'arg = {arg}')

class Foo:
    pass

print(s)
print(a)
foo('quux')
x = Foo()
print(x)
```

```
C:\Users\john\Documents>python mod.py
If Comrade Napoleon says it, it must be right.
[100, 200, 300]
arg = quux
<__main__.Foo object at 0x02F101D0>
```

但是在进行import的时候这些语句也会执行，可以使用以下的方式区分执行模块还是进行import:
```py
s = "If Comrade Napoleon says it, it must be right."
a = [100, 200, 300]

def foo(arg):
    print(f'arg = {arg}')

class Foo:
    pass

if (__name__ == '__main__'):
    print('Executing as standalone script')
    print(s)
    print(a)
    foo('quux')
    x = Foo()
    print(x)
```
在导入模块时，python会设置模块的dunder变量`__name__`，如果作为单独的脚本进行执行会设置成`__main__`。
这一特征在单元测试时比较有用。

## 重新加载模块
处于效率的原因，一个解释器会话智慧进行inmport一次，对于函数或者类定义来说比较正常，但是模块可能包含一些初始化的语句，
```py
a = [100, 200, 300]
print('a =', a)
```
```py
>>> import mod
a = [100, 200, 300]
>>> import mod
>>> import mod

>>> mod.a
[100, 200, 300]
```
在后续的导入中并没有执行`print`语句。弱国修改了模块需要重新加载，可以关掉解释器重新打开或者使用`reload()`函数。
```py
>>> import mod
a = [100, 200, 300]

>>> import mod

>>> import importlib
>>> importlib.reload(mod)
a = [100, 200, 300]
<module 'mod' from 'C:\\Users\\john\\Documents\\Python\\doc\\mod.py'>
```

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