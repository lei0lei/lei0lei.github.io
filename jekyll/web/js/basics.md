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