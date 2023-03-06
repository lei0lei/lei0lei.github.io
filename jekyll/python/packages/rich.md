---
layout: default
title: rich
permalink: /MachineLearning/packages/rich
parent: packages
grand_parent: python
has_toc: true
---

Github仓库: https://github.com/Textualize/rich

文档: https://rich.readthedocs.io/en/stable/introduction.html

安装:
```sh
python -m pip install rich
```

测试:
```sh
python -m rich
```

最简单的使用方式，用`rich`的`print`替换掉原生的。
```python
from rich import print

print("Hello, [bold magenta]World[/bold magenta]!", ":vampire:", locals())
```
输出:
```
Hello, World! 🧛
{
    '__name__': '__main__',
    '__doc__': None,
    '__package__': None,
    '__loader__': <class '_frozen_importlib.BuiltinImporter'>,
    '__spec__': None,
    '__annotations__': {},
    '__builtins__': <module 'builtins' (built-in)>,
    'print': <function print at 0x7f13bae2add0>
}
```

可以在python的REPL中安装:
```
>>> from rich import pretty
>>> pretty.install()
```

