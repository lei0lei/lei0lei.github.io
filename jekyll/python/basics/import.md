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

{: .warning }
ver: 0.0.1


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
在后续的导入中并没有执行`print`语句。如果修改了模块需要重新加载，可以关掉解释器重新打开或者使用`reload()`函数。
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
包通过`.`符号支持模块命名空间具有层级的结构，包避免了模块名之间的冲突。

直接使用文件夹层级结构就可以创建一个包，

mod1.py
```py
def foo():
    print('[mod1] foo()')

class Foo:
    pass
```

mod2.py
```py
def bar():
    print('[mod2] bar()')

class Bar:
    pass
```

```py
>>> from pkg.mod1 import foo
>>> foo()
[mod1] foo()

```

```py
>>> import pkg.mod1, pkg.mod2
>>> pkg.mod1.foo()
[mod1] foo()
>>> x = pkg.mod2.Bar()
>>> x
<pkg.mod2.Bar object at 0x033F7290>
```

```py
>>> from pkg.mod2 import Bar as Qux
>>> x = Qux()
>>> x
<pkg.mod2.Bar object at 0x036DFFD0>
```

```py
>>> from pkg import mod1
>>> mod1.foo()
[mod1] foo()

>>> from pkg import mod2 as quux
>>> quux.bar()
[mod2] bar()
```

```py
>>> import pkg
>>> pkg
<module 'pkg' (namespace)>
```
如上所示包的导入和模块的导入没什么不同，
下面的语句语法上看好像没问题，但是不会把模块放到本地的命名空间中:
```py
>>> pkg.mod1
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    pkg.mod1
AttributeError: module 'pkg' has no attribute 'mod1'
>>> pkg.mod1.foo()
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    pkg.mod1.foo()
AttributeError: module 'pkg' has no attribute 'mod1'
>>> pkg.mod2.Bar()
Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    pkg.mod2.Bar()
AttributeError: module 'pkg' has no attribute 'mod2'
```

## 包初始化

如果包的目录下有一个叫做`__init__.py`的文件，在包或者包里面的模块被import时会自动调用。这可以用来执行包的初始化代码，比如初始化包一层的数据:

init.py
```py
print(f'Invoking __init__.py for {__name__}')
A = ['quux', 'corge', 'grault']
```

![](https://files.realpython.com/media/pkg2.dab97c2f9c58.png)

```py
>>> import pkg
Invoking __init__.py for pkg
>>> pkg.A
['quux', 'corge', 'grault']
```

当导入包时会初始化全局列表`A`,包中的模块可以通过import访问全局变量：

mod1.py
```py
def foo():
    from pkg import A
    print('[mod1] foo() / A = ', A)

class Foo:
    pass
```

```py
>>> from pkg import mod1
Invoking __init__.py for pkg
>>> mod1.foo()
[mod1] foo() / A =  ['quux', 'corge', 'grault']
```

`__init__.py`还可以用来影响从一个包中对模块的自动导入，比如之前的例子中只把`pkg`放在调用者的本地符号变量而没有导入任何模块，但是如果`__init__.py`包含下列代码:

```py
print(f'Invoking __init__.py for {__name__}')
import pkg.mod1, pkg.mod2
```
可以实现:
```py
>>> import pkg
Invoking __init__.py for pkg
>>> pkg.mod1.foo()
[mod1] foo()
>>> pkg.mod2.bar()
[mod2] bar()
```
{: .note }
python3.3之后引入了隐式的命名空间包，可以没有`__init__`文件就能创建包。

## 命名空间包

https://realpython.com/python-namespace-package/


## `import *`
在之前我们看到了模块的`import *`，下面谈一下包的:

![](https://files.realpython.com/media/pkg3.d2160908ae77.png)

mod1.py
```py
def foo():
    print('[mod1] foo()')

class Foo:
    pass
```

mod2.py
```py
def bar():
    print('[mod2] bar()')

class Bar:
    pass
```

mod3.py
```py
def baz():
    print('[mod3] baz()')

class Baz:
    pass
```

mod4.py
```py
def qux():
    print('[mod4] qux()')

class Qux:
    pass
```
对于模块`import *`会跳过双下划线开头的名字，

```py
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']

>>> from pkg.mod3 import *

>>> dir()
['Baz', '__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'baz']
>>> baz()
[mod3] baz()
>>> Baz
<class 'pkg.mod3.Baz'>
```

对于包:
```py
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']

>>> from pkg import *
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']
```
没有从包里导入任何名字，python 有这样的传统，如果`__init__`文件中定义了一个`__all__`的列表，在使用`import *`时会调入这个列表中的东西。

对于上面的例子如果使用:
```py
__all__ = [
        'mod1',
        'mod2',
        'mod3',
        'mod4'
        ]
```

```py
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__']

>>> from pkg import *
>>> dir()
['__annotations__', '__builtins__', '__doc__', '__loader__', '__name__',
'__package__', '__spec__', 'mod1', 'mod2', 'mod3', 'mod4']
>>> mod2.bar()
[mod2] bar()
>>> mod4.Qux
<class 'pkg.mod4.Qux'>
```
这样对于包来说使用`import *`就不像模块一样糟糕了。在模块中也可以使用`__all__`。这样就只会import`__all__`中的名字。
总之`__all__`就是用来控制`import *`的行为的。

## 子包
包可以在任意层级上嵌入子包，

![](https://files.realpython.com/media/pkg4.a830d6e144bf.png)

这里分成了两个子包，`sub_pkg1`和`sub_pkg2`.

```py
>>> import pkg.sub_pkg1.mod1
>>> pkg.sub_pkg1.mod1.foo()
[mod1] foo()

>>> from pkg.sub_pkg1 import mod2
>>> mod2.bar()
[mod2] bar()

>>> from pkg.sub_pkg2.mod3 import baz
>>> baz()
[mod3] baz()

>>> from pkg.sub_pkg2.mod4 import qux as grault
>>> grault()
[mod4] qux()
```

在一个子包中的模块可以引用兄弟子包中的对象，比如从`mod3`引用`mod1`中的可以使用绝对导入:

pkg/sub__pkg2/mod3.py
```py
def baz():
    print('[mod3] baz()')

class Baz:
    pass

from pkg.sub_pkg1.mod1 import foo
foo()
```

```py
>>> from pkg.sub_pkg2 import mod3
[mod1] foo()
>>> mod3.foo()
[mod1] foo()
```
也可以使用相对导入:

pkg/sub__pkg2/mod3.py
```py
def baz():
    print('[mod3] baz()')

class Baz:
    pass

from .. import sub_pkg1
print(sub_pkg1)

from ..sub_pkg1.mod1 import foo
foo()
```

```py
>>> from pkg.sub_pkg2 import mod3
<module 'pkg.sub_pkg1' (namespace)>
[mod1] foo()
```

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