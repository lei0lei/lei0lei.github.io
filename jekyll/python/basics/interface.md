---
layout: default
title: interface
permalink: /python/interface
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
</details

本教程中，将会:
- 理解接口如何工作以及如何创建的
- 理解接口的在动态语言的重要性
- 实现非正式的接口
- 使用`abc.ABCMeta`和`@abc.abstractmethod`实现正式接口

# 概览
接口式一个设计类的蓝图，像类一样，接口定义了方法，但是这些方法是抽象的，抽象方法是接口定义的但是不由接口进行实现，由类进行实现，给出接口的抽象方法的具体意义。

python定义接口与其他语言不同，其他语言可能会有`interface`关键字，python没有。

# 非正式接口
在一些情况下，可能不需要严格遵守python接口规则，python的动态特征允许我们实现一个非正式的接口。一个非正式的接口是一种可以进行方法覆盖的类，但是没有严格要求。

```py
class InformalParserInterface:
    def load_data_source(self, path: str, file_name: str) -> str:
        """Load in the file for extracting text."""
        pass

    def extract_text(self, full_file_name: str) -> dict:
        """Extract text from the currently loaded file."""
        pass
```

`InformalParserInterface`定义了两个方法:`.load_data_source()`和`.extract_text()`。这些方法定义了但是没有进行实现，在从这个类进行继承创建具体的类时才会进行实现。

`InformalParserInterface`就像一个标准的类一样，需要依靠`duck typing`告诉用于这个类应该用作接口。

具体类是实现了接口中方法的子类，
```py
class PdfParser(InformalParserInterface):
    """Extract text from a PDF"""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides InformalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides InformalParserInterface.extract_text()"""
        pass

class EmlParser(InformalParserInterface):
    """Extract text from an email"""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides InformalParserInterface.load_data_source()"""
        pass

    def extract_text_from_email(self, full_file_path: str) -> dict:
        """A method defined only in EmlParser.
        Does not override InformalParserInterface.extract_text()
        """
        pass
```
现在我们定义了接口的两个具体实现，但是`EmlParser`没有合适的定义`.extract_text()`.要进行检查的话，

```py
>>> # Check if both PdfParser and EmlParser implement InformalParserInterface
>>> issubclass(PdfParser, InformalParserInterface)
True

>>> issubclass(EmlParser, InformalParserInterface)
True
```
这可能会出现问题，因为EmlParser没有完整实现接口。可以检查两个子类的mro(method resolution order)：

```py
>>> PdfParser.__mro__
(__main__.PdfParser, __main__.InformalParserInterface, object)

>>> EmlParser.__mro__
(__main__.EmlParser, __main__.InformalParserInterface, object)
```

如果我们想要`issubclass(EmlParser, InformalParserInterface)`返回False,可以创建`metaclass`来覆盖两个dunder方法:

1. `.__instancecheck__()`
2. `.__subclasscheck__()`

```py
class ParserMeta(type):
    """A Parser metaclass that will be used for parser class creation.
    """
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text))

class UpdatedInformalParserInterface(metaclass=ParserMeta):
    """This interface is used for concrete classes to inherit from.
    There is no need to define the ParserMeta methods as any class
    as they are implicitly made available via .__subclasscheck__().
    """
    pass
```
下面是具体的实现:

```py
class PdfParserNew:
    """Extract text from a PDF."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides UpdatedInformalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides UpdatedInformalParserInterface.extract_text()"""
        pass
```

`PdfParserNew`重写了两个方法因此`issubclass(PdfParserNew, UpdatedInformalParserInterface)`返回True.

使用metaclass，不需要明确的定义子类，子类必须定义对应的方法否则`issubclass(EmlParserNew, UpdatedInformalParserInterface)`会返回False.

```py
>>> issubclass(PdfParserNew, UpdatedInformalParserInterface)
True

>>> issubclass(EmlParserNew, UpdatedInformalParserInterface)
False
```
现在看一下MRO：

```py
>>> PdfParserNew.__mro__
(<class '__main__.PdfParserNew'>, <class 'object'>)
```

可以看出`UpdatedInformalParserInterface`是`PdfParserNew`的父类，但是并没有出现在MRO中，因为`UpdatedInformalParserInterface`是`PdfParserNew`。

这种写法和标准的子类方式的区别在于虚基类使用了`.__subclasscheck__()`dunder方法来隐式的检查一个类是否是父类的虚拟子类，另外虚拟基类不会出现在子类的MRO中。

```py
class PersonMeta(type):
    """A person metaclass"""
    def __instancecheck__(cls, instance):
        return cls.__subclasscheck__(type(instance))

    def __subclasscheck__(cls, subclass):
        return (hasattr(subclass, 'name') and 
                callable(subclass.name) and 
                hasattr(subclass, 'age') and 
                callable(subclass.age))

class PersonSuper:
    """A person superclass"""
    def name(self) -> str:
        pass

    def age(self) -> int:
        pass

class Person(metaclass=PersonMeta):
    """Person interface built from PersonMeta metaclass."""
    pass
```

1. metaclass `PersonMeta`
2. 基类 `PersonSuper`
3. 接口 `Person`

创建完虚基类之后可以定义两个具体类，`Employee`继承自`PersonSuper`，`Friend`隐式的继承自`Person`.

```py
# Inheriting subclasses
class Employee(PersonSuper):
    """Inherits from PersonSuper
    PersonSuper will appear in Employee.__mro__
    """
    pass

class Friend:
    """Built implicitly from Person
    Friend is a virtual subclass of Person since
    both required methods exist.
    Person not in Friend.__mro__
    """
    def name(self):
        pass

    def age(self):
        pass
```
尽管`Friend`没有明确的继承自`Person`,它实现了`.name()`和`.age()`，因此变成了`Friend`的虚基类，在运行`issubclass(Friend, Person)`的时候应该返回True.

![UML](https://files.realpython.com/media/virtual-base-class.b545144aafef.png)

Taking a look at PersonMeta, you’ll notice that there’s another dunder method called .__instancecheck__(). This method is used to check if instances of Friend are created from the Person interface. Your code will call .__instancecheck__() when you use isinstance(Friend, Person).

查看一下`PersonMeta`,将会发现由另一个dunder方法`__instancecheck__()`,这个方法用来检查Friend的实力是否是从`Person`接口创建的实例。在使用`isinstance(Friend, Person)`的时候会调用该方法。

# 正式接口
开发者较少时非正式的接口比较有用，但是对于打的应用需要创建正式的python接口，这需要从`abc`模块中拿一些工具来用。

## 使用`abc.ABCMeta`

为了强制子类实例化抽象方法，需要利用内建的`abc`模块中的`ABCMeta`. 回到之前的`UpdatedInformalParserInterface`接口，我们创建了自己的metaclass：`ParserMeta`，重写了dunder方法:`.__instancecheck__()`和`.__subclasscheck__()`。

除了自己创建metaclass,可以使用`abc.ABCMeta`作为metaclass.然后重写`.__subclasshook__()`替代`.__instancecheck__()`和`.__subclasscheck__()`,subclasshook的实现更加可靠。

## 使用`.__subclasshook__()`
下面是使用`abc.ABCMeta`作为metaclass的`FormalParserInterface`实现。

```py
import abc

class FormalParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text))

class PdfParserNew:
    """Extract text from a PDF."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides FormalParserInterface.extract_text()"""
        pass

class EmlParserNew:
    """Extract text from an email."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text_from_email(self, full_file_path: str) -> dict:
        """A method defined only in EmlParser.
        Does not override FormalParserInterface.extract_text()
        """
        pass
```
如果在`PdfParserNew`和`EmlParserNew`上运行`issubclass()`，将会返回True和False.

## 使用`abc`注册虚拟子类

import `abc`模块之后，可以直接使用`.register()`元方法注册虚拟子类，这一个例子注册了一个`Doblue`接口作为内建的`__float__`类的虚拟基类。

```py
class Double(metaclass=abc.ABCMeta):
    """Double precision floating point number."""
    pass

Double.register(float)
```

可以使用`.register()`检查效果，

```py
>>> issubclass(float, Double)
True

>>> isinstance(1.2345, Double)
True
```

通过使用`.register()`元方法，注册了一个`float`的虚拟子类`Double`.

可以使用类装饰其将被装饰的类设置为虚拟子类，

```py
@Double.register
class Double64:
    """A 64-bit double-precision floating-point number."""
    pass

print(issubclass(Double64, Double))  # True
```

### 使用带注册的子类检测

在结合使用`.__subclasshook__()`和`.register()`的时候必须小心，因为`.__subclasshook__()`的优先级高于虚拟子类注册。为了确保注册的虚拟子类生效必须给`.__subclasshook__()`方法添加`NotImplemented`.更新后的`FormalParserInterface`为:

```py
class FormalParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text) or 
                NotImplemented)

class PdfParserNew:
    """Extract text from a PDF."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides FormalParserInterface.extract_text()"""
        pass

@FormalParserInterface.register
class EmlParserNew:
    """Extract text from an email."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text_from_email(self, full_file_path: str) -> dict:
        """A method defined only in EmlParser.
        Does not override FormalParserInterface.extract_text()
        """
        pass

print(issubclass(PdfParserNew, FormalParserInterface))  # True
print(issubclass(EmlParserNew, FormalParserInterface))  # True
```

`EmlParserNew`被认为是`FormalParserInterface`接口的虚拟子类。这出现了一点问题因为`EmlParserNew`没有覆盖`.extract_text()`.使用虚拟子类注册一定要小心。

### 使用抽象方法声明

抽象方法是python接口声明的方法，但是没有一个有用的实现。抽象方法必须被实现了接口的具体类进行重写。

要在python中创建抽象方法，将`@abc.abstractmethod`装饰其添加到接口的方法中，在下一个例子中，更改`FormalParserInterface`来包含抽象方法`.load_data_source()`和`.extract_text()`.

```py
class FormalParserInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'load_data_source') and 
                callable(subclass.load_data_source) and 
                hasattr(subclass, 'extract_text') and 
                callable(subclass.extract_text) or 
                NotImplemented)

    @abc.abstractmethod
    def load_data_source(self, path: str, file_name: str):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def extract_text(self, full_file_path: str):
        """Extract text from the data set"""
        raise NotImplementedError

class PdfParserNew(FormalParserInterface):
    """Extract text from a PDF."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_path: str) -> dict:
        """Overrides FormalParserInterface.extract_text()"""
        pass

class EmlParserNew(FormalParserInterface):
    """Extract text from an email."""
    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides FormalParserInterface.load_data_source()"""
        pass

    def extract_text_from_email(self, full_file_path: str) -> dict:
        """A method defined only in EmlParser.
        Does not override FormalParserInterface.extract_text()
        """
        pass
```
在上边的例子中，最终创建了一个正式接口，当抽象方法没有被重写时会报错。

`PdfParserNew`实例`pdf_parser`不会报错，因为`PdfParserNew`正确重写了`FormalParserInterface`的抽象方法，但是`EmlParserNew`会报错:

```py
>>> pdf_parser = PdfParserNew()
>>> eml_parser = EmlParserNew()
Traceback (most recent call last):
  File "real_python_interfaces.py", line 53, in <module>
    eml_interface = EmlParserNew()
TypeError: Can't instantiate abstract class EmlParserNew with abstract methods extract_text
```
回溯消息告诉我们没有覆盖所有的抽象方法，这就是一个正式的python接口。