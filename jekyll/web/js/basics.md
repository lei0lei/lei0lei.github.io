---
layout: default
title: basics
permalink: /web/js/basics
parent: js
grand_parent: web
has_toc: true
---

# hello world

## The “script” tag
使用script tag可以将javascript代码嵌入到html页面的几乎所有地方。比如:
```html
<!DOCTYPE HTML>
<html>
<body>
  <p>Before the script...</p>
  <script>
    alert( 'Hello, world!' );
  </script>
  <p>...After the script.</p>
</body>
</html>
```


## 现代标记语言

`<script>` tag有一些比较过时的属性:

- `<script type=…>`,html4中需要脚本具有类型，通常是`type="text/javascript"`.但是现在不再需要了，现代的html完全改变了这个属性的意义，现在可以用于js模块。

- `<script language=…>`这个属性用来显式脚本的语言，由于js是默认语言，因此现在没必要使用了。

比较旧的代码中可能发现:
```html
<script type="text/javascript"><!--
    ...
//--></script>
```
但是现在不再使用了。
## 外部脚本
如果有很多的代码可以把js单独放在一个文件中，然后使用`src`属性连接到html:
```html
<script src="/path/to/script.js"></script>
```

`/path/to/script.js` 是脚本相对于站点root的绝对路径。也可以使用相对当前页面的路径，比如`src="script.js"`,就像`src="./script.js"`,意味着当前路径下的文件"script.js".
也可以使用url：
```js
<script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.11/lodash.js"></script>
```
要加入多个脚本:
```js
<script src="/js/script1.js"></script>
<script src="/js/script2.js"></script>
```

{: .note :}
> 注意:
>
> 应该只把最简单的脚本加到html中，复杂脚本放到单独文件中，单独的文件使得浏览器可以下载放到缓存中。其他页面引用的时候直接从缓存中取而不是再下载。

{: .warning :}
> 设置了`src`后会忽视脚本内容
>
> 单独的`<script>` tag不能同时有`src`属性和代码，比如
> ```js
> <script src="file.js">
>  alert(1); // the content is ignored, because src is set
> </script>
> ```
> 会忽略掉代码。

# 代码结构

## 语句
语句可以使用分号或者断行进行隔开，
```js
alert('Hello')
alert('World')
```
js不会在`[...]`之前假设有分号，可以尝试以下代码:
```js
alert("Hello")

[1, 2].forEach(alert);
```

```js
alert("Hello");

[1, 2].forEach(alert);
```

## 注释

单行注释使用`//`.
```js
// This comment occupies a line of its own
alert('Hello');

alert('World'); // This comment follows the statement
```

多行注释使用`/*...*/`.
```js
/* An example with two messages.
This is a multiline comment.
*/
alert('Hello');
alert('World');
```

不支持嵌套注释比如:
```js
/*
  /* nested comment ?!? */
*/
alert( 'World' );
```

# modern mode

很长时间依赖js没有兼容性问题，这可以保证现存的代码运行起来没问题，但是问题是语言开发者可能导致语言出现问题，在ES5出现后给语言加入了一些新的特征，要使旧的代码运行，需要使用声明`"use strict"`.

```js
"use strict";

// this code works the modern way
...
```
这句话出现在脚本最开头的时候表示整个脚本以现代的方式运行。这句话也可以用在函数里只对函数起作用。

{: .warning :}
> 确保`'use strict'`出现在脚本开头，否则不会生效。

使用浏览器控制台时默认不会使用`use strict`. 应该摁`Shift+Enter`输入多行，如果在旧的浏览器不行的话使用:
```js
(function() {
  'use strict';

  // ...your code here...
})()
```
现代js支持`class`以及`modules`会自动启用`use strict`

# 变量


# 数据类型


# 类型转换


# 交互

# 算子

# 比较

# 条件

# 逻辑算符

# `？？`

# 循环

# switch

# 函数

# 箭头函数

# js specials