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

# [OOP](https://realpython.com/python3-object-oriented-programming/)
面向对象编程是一个编程范式，将属性和行为绑定到独立的对象上。
对应的另一种常见的范式是过程式编程。
## 定义类
基础的数据结构像number,string以及list，用于表示简单的信息，如果需要表示一些复杂的东西比如追踪一个人员在组织中的行为，需要保存该人员的基础信息比如姓名，年龄，职位，工作年龄，一种方式是使用list表示:

```py
kirk = ["James Kirk", 34, "Captain", 2265]
spock = ["Spock", 35, "Science Officer", 2254]
mccoy = ["Leonard McCoy", "Chief Medical Officer", 2266]
```
这种方法存在很多问题，大的代码文件很难管理，如果在其他地方引用kirk很难记住kirk的实际定义。第二，如果每个人员的元素数量并不相同就会产生错误，比如`mccoy`.更好的替代方式就是使用类来方便管理和维护。

### 类vs实例
类用于创建用户自定义的数据结构，类定义的函数叫做方法，定义了从该类创建的对象操作其数据的行为和方法。

类是一个蓝图表示某个东西应该如何定义，不包含实际的数据，比如一个`Dog`类并不包含实际的狗的年龄和名字。实际包含数据的是类实例化出来的对象。

### 如何定义类
类的定义使用`class`关键字，后面跟着类的名字和冒号。下面是一个例子:

```py
class Dog:
    pass
```
`Dog`类的body包含一个语句:pass语句，这是一个占位符。让我们给这个类加一些内容，首先所有的`Dog`对象都必须要定义一个叫做`.__init__()`的方法，每次一个新的对象创建，`.__init__()`都会给属性进行赋值来设置对象的初始状态。`.__init__()`可以包含任意数量的参数，但是第一个参数必须叫做`self`.当创建新的类实例的时候，实例自动传给`self`参数，这样才可以定义新的属性。

```py
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```
在`.__init__()`中创建的属性叫做实例属性，实例的属性对不同的对象是不同的可以在`__init__()`之外给一个变量命名。类的属性对所有的类实例相同，在`__init__()`之外可以定义类属性，比如:

```py
class Dog:
    # Class attribute
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age
```
类的属性必须有初始值，在实例创建的时候，类属性会自动创建和赋值。

## 实例化类
创建对象的过程叫做对象的实例化，

```py
>>> Dog()
<__main__.Dog object at 0x106702d30>
```
现在在内存地址`0x106702d30`处有了新的对象，再来创建一个新的对象，

```py
>>> a = Dog()
>>> b = Dog()
>>> a == b
False
```
如果使用`==`操作符比较两个对象会发现结果是`False`，因为表示内存中的不同对象。

### 类和实例属性
```py
class Dog:
    species = "Canis familiaris"
    def __init__(self, name, age):
        self.name = name
        self.age = age
```
要实例化一个对象需要给`name`和`age`一个值:

```py
>>> Dog()
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    Dog()
TypeError: __init__() missing 2 required positional arguments: 'name' and 'age'
```
要初始化直接放在括号里即可:
```py
buddy = Dog("Buddy", 9)
miles = Dog("Miles", 4)
```
这会创建两个实例，在实例化`Dog`对象的时候，python 会创建一个新的实例然后传递给`.__init__()`的第一个参数。

创建了实例后可以通过`.`写法来进行属性访问：

```py
>>> buddy.name
'Buddy'
>>> buddy.age
9

>>> miles.name
'Miles'
>>> miles.age
4
```
可以以相同的方式访问类的属性:

```py
>>> buddy.species
'Canis familiaris'
```
使用类来组织数据的优点是实例保证会有期望的属性，因此总会返回一个值。

尽管属性一定会存在，其值可以动态的改变:

```py
>>> buddy.age = 10
>>> buddy.age
10

>>> miles.species = "Felis silvestris"
>>> miles.species
'Felis silvestris'
```
自定义个对象默认上是`mutable`的，换句话说，可以被动态改变，比如lists和dict都是`mutable`，但是string和tuple是`immutable`.
### 实例方法
实例方法是在类里面定义的函数并且只能从类的实力进行调用，实例方法的第一个参数都是`self`,

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Instance method
    def description(self):
        return f"{self.name} is {self.age} years old"

    # Another instance method
    def speak(self, sound):
        return f"{self.name} says {sound}"
```

`Dog`类有两个实例方法:

- `.description()`返回一个字符串显示狗的name和age.
- `.speak()`有一个叫做sound的参数并返回name和sound.

```py
>>> miles = Dog("Miles", 4)

>>> miles.description()
'Miles is 4 years old'

>>> miles.speak("Woof Woof")
'Miles says Woof Woof'

>>> miles.speak("Bow Wow")
'Miles says Bow Wow'
```
在写自己的类的时候，包含一个方法来返回这个类的一些信息通常是有用的，但是`.description()`不是那么的`Pythonic`.

当我们`print`一个类的时候:

```py
>>> print(miles)
<__main__.Dog object at 0x00aeff70>
```
返回的信息并没有什么用，通过改变一个特殊的实例方法`.__str__()`可以改变这个行为:

```py
class Dog:
    # Leave other parts of Dog class as-is

    # Replace .description() with __str__()
    def __str__(self):
        return f"{self.name} is {self.age} years old"
```

```py
>>> miles = Dog("Miles", 4)
>>> print(miles)
'Miles is 4 years old'
```

前后置双下下划线在python中叫做`dunder methods`。

## 继承

继承允许一个类得到另一个类的属性和方法，新类叫做子类。子类可以覆盖或者扩展父类的属性和方法，换句话说，子类继承了父类的所有属性和方法但同时实现了自己的属性和方法。

比如我们要给狗加入一个品种的属性区分不同的狗，一种方式是直接添加属性，这样既需要在实例化的时候额外传入一个参数；

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age, breed):
        self.name = name
        self.age = age
        self.breed = breed
```
```py
>>> miles = Dog("Miles", 4, "Jack Russell Terrier")
>>> buddy = Dog("Buddy", 9, "Dachshund")
>>> jack = Dog("Jack", 3, "Bulldog")
>>> jim = Dog("Jim", 5, "Bulldog")
```

但是每只品种的狗有不同的行为，比如斗牛叫声是woof,dachshund叫声是yap,如果只使用一个类每次就需要给speak额外添加一个参数,这会很麻烦，而且一只狗的叫声应该取决于其品种。

```py
>>> buddy.speak("Yap")
'Buddy says Yap'

>>> jim.speak("Woof")
'Jim says Woof'

>>> jack.speak("Woof")
'Jack says Woof'
```

创建一个子类可以解决这个问题，每个子类可以继承speak功能进行改造,包括为每个子类定义一个默认参数。

### 父类 VS 子类

```py
class Dog:
    species = "Canis familiaris"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"{self.name} is {self.age} years old"

    def speak(self, sound):
        return f"{self.name} says {sound}"

class JackRussellTerrier(Dog):
    pass

class Dachshund(Dog):
    pass

class Bulldog(Dog):
    pass
```

要创建子类只需要将父类名字放在括号中。

```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> buddy = Dachshund("Buddy", 9)
>>> jack = Bulldog("Jack", 3)
>>> jim = Bulldog("Jim", 5)
```
子类的实例继承了父类的所有属性和方法:
```py
>>> miles.species
'Canis familiaris'

>>> buddy.name
'Buddy'

>>> print(jack)
Jack is 3 years old

>>> jim.speak("Woof")
'Jim says Woof'
```

要确定对象属于哪个类只需要使用内建的`type()`：

```py
>>> type(miles)
<class '__main__.JackRussellTerrier'>
```
如果想要知道某个对象是否是类的实例只需要使用内建的`isinstance()`:

```py
>>> isinstance(miles, Dog)
True
```

`isinstance()`有两个参数，第一个是对象，第二个是类。

```py
>>> isinstance(miles, Bulldog)
False

>>> isinstance(jack, Dachshund)
False
```
从子类创建的对象都是父类的实例。

### 扩展父类的功能

由于不同品种的狗不同的叫声，可以为子类的`.speak()`方法提供一个默认值，只需要覆盖`.speak()`方法。

```py
class JackRussellTerrier(Dog):
    def speak(self, sound="Arf"):
        return f"{self.name} says {sound}"
```
要覆盖父类的方法只需要在子类中命名相同的方法:

```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> miles.speak()
'Miles says Arf'
```
有的时候我们会需要完全覆盖父类的方法，但是有时我们需要保留父类的东西，可以在子类中哦功能调用父类的方法,使用`super()`：

```py
class JackRussellTerrier(Dog):
    def speak(self, sound="Arf"):
        return super().speak(sound)
```
调用`super()`的时候python回去搜索父类`Dog`中的`.speak方法。
```py
>>> miles = JackRussellTerrier("Miles", 4)
>>> miles.speak()
'Miles barks: Arf'
```

# [实例、类和静态方法](https://realpython.com/instance-class-and-static-methods-demystified/)



# [重载](https://realpython.com/operator-function-overloading/)


# [属性](https://realpython.com/python-property/)



# [继承和多态](https://realpython.com/inheritance-composition-python/)



# [dataclass](https://realpython.com/python-data-classes/)



# [构造器](https://realpython.com/python-class-constructor/)



# [多重构造器](https://realpython.com/python-multiple-constructors/)




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

新风格的类统一了类和类型的概念。如果`obj`是新风格的类的实例，`type(obj)`和`obj.__class__`相同:

```py
>>> class Foo:
...     pass
>>> obj = Foo()
>>> obj.__class__
<class '__main__.Foo'>
>>> type(obj)
<class '__main__.Foo'>
>>> obj.__class__ is type(obj)
True
```

```py
>>> n = 5
>>> d = { 'x' : 1, 'y' : 2 }

>>> class Foo:
...     pass
...
>>> x = Foo()

>>> for obj in (n, d, x):
...     print(type(obj) is obj.__class__)
...
True
True
True
```

## 类型和类