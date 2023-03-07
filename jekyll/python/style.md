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
| 类型 | 命名 |  样例|
|:--|:--|:--|
|函数|使用小写，下划线分割| my_function|
|变量|使用小写，下划线分割|x,var,my_variable|
|类|首字母大写，不要使用下划线|MyClass|
|方法|小写，下划线分割|class_method|
|常量|纯大写，下划线分割|MY_CONSTANT|
|模块|小写，下划线分割|my_module.py|
|包|小写，不要使用下划线|mypackage|

- 使用描述性名字
- 不要使用无意义的名字如x,y除非是在用数学函数
- 不要使用歧义缩写



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

下面是一个docstring的例子:

```py
def square(n):
    '''Takes in a number n, returns the square of n'''
    return n**2
```
python docstring是用在函数、方法、类、或者模块中的字符串来对代码进行文档化，要进行访问，
使用`__doc__`属性。

## __doc__属性

当字符串放在函数、模块、方法或者类的定义后边时就会与对象的`__doc__`属性关联，
可以通过这个属性进行访问。比如:

```py
def square(n):
    '''Takes in a number n, returns the square of n'''
    return n**2

print(square.__doc__)
```

### 单行docstring

- 单行docstring只有一行，前后中间都没有空行。
- 不应该是描述性的必须遵循"Do this, return that" 类似的结构
```py
def multiplier(a, b):
    """Takes in two numbers, returns their product."""
    return a*b
```
### 多行docstring

多行docstring包含一个summary，类似单行。后边跟一个空行以及更加
详细的描述，具体查看:PEP 257.
#### Python模块中的docstring

  - 列出所有的类函数和对象以及ecxcepions
  - 每一项都应该改有一行总结
下面是一个例子:

```
Create portable serialized representations of Python objects.

See module copyreg for a mechanism for registering custom picklers.
See module pickletools source for extensive comments.

Classes:

    Pickler
    Unpickler

Functions:

    dump(object, file)
    dumps(object) -> string
    load(file) -> object
    loads(string) -> object

Misc variables:

    __version__
    format_version
    compatible_formats
```
#### Python类中的docstring

- 应该给出类的行为以及其公有方法和实例变量
- 子类，构造器和方法应该有自己的docstring

```py
class Person:
    """
    A class to represent a person.

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(self, name, surname, age):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """

        self.name = name
        self.surname = surname
        self.age = age

    def info(self, additional=""):
        """
        Prints the person's name and age.

        If the argument 'additional' is passed, then it is appended after the main info.

        Parameters
        ----------
        additional : str, optional
            More info to be displayed (default is None)

        Returns
        -------
        None
        """

        print(f'My name is {self.name} {self.surname}. I am {self.age} years old.' + additional)
```
可以使用`help()`函数查看对象对应的docstring.
#### python函数中的docstring
- 应该给整个函数一个总结以及器参数和返回值
- 应该列出所有的exceptions和可选参数
下面是一个例子:

```py
def add_binary(a, b):
    '''
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    a (int): A decimal integer
                    b (int): Another decimal integer

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    '''
    binary_sum = bin(a+b)[2:]
    return binary_sum


print(add_binary.__doc__)
```
#### python脚本的docstring

- 应该给出脚本的函数和命令行语法使用
- 应该作为所有函数和参数的快速索引

#### Python包中的docstring

- 应该写在`__init__.py`文件中
- 应该包含包可以导出的所有模块和子包

## docstring格式
### Epytest

```py
"""
This is a javadoc style.

@param param1: this is a first param
@param param2: this is a second param
@return: this is a description of what is returned
@raise keyError: raises an exception
"""
```
### reST

```py
"""
This is a reST style.

:param param1: this is a first param
:param param2: this is a second param
:returns: this is a description of what is returned
:raises keyError: raises an exception
"""
```
### Google

```py
"""
This is an example of Google style.

Args:
    param1: This is the first param.
    param2: This is a second param.

Returns:
    This is a description of what is returned.

Raises:
    KeyError: Raises an exception.
"""
```
### Numpydoc

```py
"""
My numpydoc description of a kind
of very exhautive numpydoc format docstring.

Parameters
----------
first : array_like
    the 1st param name `first`
second :
    the 2nd param
third : {'value', 'other'}, optional
    the 3rd param, by default 'value'

Returns
-------
string
    a value in a string

Raises
------
KeyError
    when a key error
OtherError
    when an other error
"""
```
可以使用Pyment自动给一个项目生成docstring 

## 代码布局

- 空行，空格提升可读性，顶层函数和类使用两个空行，内部方法使用单空行，不同步骤之间加空行。
- 每一行限制到79个字符，

```py
def function(arg_one, arg_two,
             arg_three, arg_four):
    return arg_one
```

```py
from mypkg import example1, \
    example2, example3
```

```py
# Recommended
total = (first_variable
         + second_variable
         - third_variable)
```

## 缩进

- 使用4个连续space表示缩进
- 使用space而不是tab
- 断行之后使用缩进

```py
def function(arg_one, arg_two,
             arg_three, arg_four):
    return arg_one
```

```py
x = 5
if (x > 3 and
        x < 10):
    print(x)
```

```py
var = function(
    arg_one, arg_two,
    arg_three, arg_four)

def function(
        arg_one, arg_two,
        arg_three, arg_four):
    return arg_one
```
下面这种写法是不被推荐的，使用hanging indent时尽量不要在第一行留参数:

```py
# Not Recommended
var = function(arg_one, arg_two,
    arg_three, arg_four)
```

- 在断行的情况下如何使用括号
```py
list_of_numbers = [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
    ]

list_of_numbers = [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
]
```

## 注释
- 注释和docstring每行不超过72字符
- 使用完整语句，大写字母开头
- 更改代码后更新注释
- #space 开头，缩进与代码对齐，分段使用#多空一行

```py
def quadratic(a, b, c, x):
    # Calculate the solution to a quadratic equation using the quadratic
    # formula.
    #
    # There are always two solutions to a quadratic equation, x_1 and x_2.
    x_1 = (- b+(b**2-4*a*c)**(1/2)) / (2*a)
    x_2 = (- b-(b**2-4*a*c)**(1/2)) / (2*a)
    return x_1, x_2
```
- 尽量少用内联的注释
- 内敛注释和正式代码有至少两个空格

## 表达式和语句空白

- 赋值运算符，比较运算符和逻辑运算符两端有空白
- 函数参数礼的默认值等号两边不要有空白

```py
# Recommended
y = x**2 + 5
z = (x+y) * (x-y)

# Not Recommended
y = x ** 2 + 5
z = (x + y) * (x - y)
```
- 切片运算符两边有空白
- 千万不要在每一行后边添加空白，下面是一些不应该添加空白的地方

```py
# Recommended
my_list = [1, 2, 3]

# Not recommended
my_list = [ 1, 2, 3, ]
x = 5
y = 6

# Recommended
print(x, y)

# Not recommended
print(x , y)

def double(x):
    return x * 2

# Recommended
double(3)

# Not recommended
double (3)

# Recommended
list[3]

# Not recommended
list [3]

# Recommended
tuple = (1,)

# Not recommended
tuple = (1, )

# Recommended
var1 = 5
var2 = 6
some_long_var = 7

# Not recommended
var1          = 5
var2          = 6
some_long_var = 7
```

