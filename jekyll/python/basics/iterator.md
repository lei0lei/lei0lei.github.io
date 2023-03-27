---
layout: default
title: iterator
permalink: /python/iterator
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

https://realpython.com/python-iterators-iterables/

# 基础
`iterators`和`iterables`是两种不同但是相关的工具可以用来在一个数据流或者容器内进行迭代。iterator控制着迭代过程，iterable存放这数据内容。

iterator和iterable是python的基础组件，绝大多数程序中都会遇到。本文章包含:

- 使用`iterator protocol`创建iterators
- 理解iterator和iterable的不同
- 使用iterator和iterable
- 使用`generator functions`和`yield`语句创建`generator iterator`
- 使用不同的技术构建自己的iterable
- 使用asyncio模块以及await和asynv关键字创建异步iterator