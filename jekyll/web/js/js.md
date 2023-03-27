---
layout: default
title: js
permalink: /web/js
parent: web
has_children: true
has_toc: true
---



# What is JavaScript?
JavaScript 最开始的作用是使得web页面'活起来'.这种语言中的程序叫做脚本，可以直接在页面的html中写然后在页面加载的时候自动运行。

{: .note :}
> 为何叫做javascript?
> 
> 最开始的时候叫做Livescript，但是由于当时java很火，因此为了营销...
> 后来javascript变成完全独立的语言，有自己的specification叫做ECMAScript.

现在javascript不仅可以在浏览器中运行也能在服务器上运行或者在有js引擎的机器上运行。浏览器里又一个叫做javascript虚拟机的嵌入式引擎，不同的引擎有不同的名字:
- v8, chrome opera edge
- spidermonkey, firefox
- chakra, ie
- javascriptcore,nitro,squirrelfish safari

{: .note :}
> 引擎如何工作的？
> 
> 1. 引擎读取脚本
> 2. 将脚本转化(编译)为机器码
> 3. 机器码非常快的运行

> 引擎对这个过程的每一步进行优化，在编译好的脚本运行的时候甚至都会进行监控分析数据流然后进一步优化机器码

# 浏览器内javascript可以做什么?
现代js是一个"安全"的编程语言，不提供对内存或者cpu的底层访问，因为为浏览器写的不需要这样做。js的能力很大程度上依赖于运行的环境，比如nodejs支持js读写任何文件，发起网络请求等，浏览器内的js可以做到和web页面相关的错做，和用户以及webserver进行交互.比如:
- 给页面添加新的html,更改现在的内容或者风格
- 响应用户的动作，运行鼠标点击移动键盘按键等
- 通过网络向远程服务器发送请求，下载和上传文件(AJAX,COMET)
- 获取或者设置cookies,显示信息，向访问者发送问题
- 记住客户端数据

# 浏览器内javascript做不到什么?
在浏览器中的javascript受到限制来保护用户的安全，防止有害的页面获取隐私信息或者危害用户的数据。比如下面这些无法做到的:
- 读写硬盘上的任意文件，拷贝或者执行文件。无法直接访问系统函数
- 现代浏览器运行处理文件但是访问受限并且在用户允许的条件下才行，比如将文件拖动到浏览器或者通过<input> tag进行选择
- 可以和相机mac或者其他设备进行交互但是明确需要用户的权限。
- 不同的tab之间一般是隔离的，有时js可以打开另一个窗口但是这种情况下也无法访问来自另一个网站的页面(不同的域名协议或者端口),这叫做"same origin policy"，要想实现两个页面必须同意信息交互，并包含特定的js代码来处理
- js很容易通过网络和服务器进行交互，但是从另外的网站或域名接收信息是受限的，当然并不是不可能，需要远程在http header进行特殊操作。

如果在浏览器外比如服务器上，这种限制就不存在，现代浏览器允许申请额外的权限。

javascript有一些很好的特征:
- 完全集成html/css
- Simple things are done simply.
- 默认支持所有主流浏览器


# js "之外"
js的语法无法满足所有人的需要，不同的人可能需要不同的特征。于是出先了一些新的语言可以在浏览器中运行之前转换成javascript.现代工具可以使得转换非常迅速进行自动转换，下面是一些其他语言:

- CoffeeScript JavaScript语法糖，语法更短, ruby开发这会比较喜欢.
- TypeScript 添加了严格数据类型来简化开发和支持复杂系统，由microsoft开发.
- Flow 添加了数据类型但是方式不同，facebook开发.
- Dart 一个独立的语言，有自己的浏览器外环境运行的引擎但是可以转换成js，google开发.
- Brython 可以用python开发而不用js.
- Kotlin 现代化简明并安全的语言

类似的还有许多其他的语言，但是本质上我们必须了解js.
