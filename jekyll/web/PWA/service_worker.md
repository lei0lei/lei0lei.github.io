---
layout: default
title: service-worker
permalink: /web/pwa/service-worker
parent: PWA
grand_parent: web
has_toc: true
---
service worker是pwa的关键组件，必不可少。

# 介绍
service worker是一类特殊的web worker，用于应用和网络之间的代理，经过PWA的请求都会首先通过service worker.这样可以在网络连接失败的时候处理请求。

service worker是由事件驱动的运行在应用主线程之外。service worker不会阻塞用户接口，而是监听一些事件(比如`fetch`事件)，并进行异步处理。

一些使用功能如下：
- 预先缓存assets
- 处理asset请求，比如决定合适使用缓存合适使用网络
- 处理web notification相关的事件比如notification click和push event.
- 在后台中同步应用.

# 创建worker
要将service worker添加到项目，创建一个`sw.js`文件在项目的根目录下。这样可以使应用进行安装，但是缺少离线功能。

## 作用域

可以将service worker放在项目的任何地方，但是它之恩那个访问在当前目录层级之下或之中的asset。放在根目录下serrvice就会有整个应用的作用域。

## 注册
放好`sw.js`文件之后，可以在应用的index中注册service worker. 将写列内容加入`index.html`.

```html
<script>
  if (typeof navigator.serviceWorker !== 'undefined') {
    navigator.serviceWorker.register('sw.js')
  }
</script>

```

## 示例worker
### 在`install`事件中缓存
第一次发出`install`事件使会安装service worker.可以添加一个listener来预先缓存重要资源以供离线使用。
```js
const CACHE_NAME = 'cool-cache';

// Add whichever assets you want to pre-cache here:
const PRECACHE_ASSETS = [
    '/assets/',
    '/src/'
]

// Listener for the install event - pre-caches our assets list on service worker install.
self.addEventListener('install', event => {
    event.waitUntil((async () => {
        const cache = await caches.open(CACHE_NAME);
        cache.addAll(PRECACHE_ASSETS);
    })());
});
```
收到`install`时间后，这个代码会打开一个缓存叫做`CACHE_NAME`并预先缓存`PRECACHE_ASSETS`中的任何assets.

### 在`Activate`事件期间声明客户端

在install事件之后，service worker生命周期的下一步是activation.`Activation`事件会在安装完成后立刻触发。
可以使用这个事件让我们的service worker控制已经运行的app实例，又叫做`claiming a client`:
```js
self.addEventListener('activate', event => {
  event.waitUntil(clients.claim());
});

```
默认情况下新激活的service worker在重新加载之前不会claim任何客户端。用`clients.claim()`告诉service worker立刻控制新的客户端。

### 定义`Fetch`策略
在service worker预先缓存assets时，需要一些功能来检索这些assets.

监听`fetch`事件可以让我们拦截并处理assets请求。

```js
self.addEventListener('fetch', event => {
  event.respondWith(async () => {
      const cache = await caches.open(CACHE_NAME);

      // match the request to our cache
      const cachedResponse = await cache.match(event.request);

      // check if we got a valid response
      if (cachedResponse !== undefined) {
          // Cache hit, return the resource
          return cachedResponse;
      } else {
        // Otherwise, go to the network
          return fetch(event.request)
      };
  });
});

```

上面的代码显示了fetch资源的`Cache-First`策略，service worker拦截了一个请求之后会首先检查缓存，没找到的话才会连接网络。

### 安全性

service worker必须从安全的启用了https的endpoint来提供服务，为了测试的需求，service worker可以使用localhost,但是如果想要进行分发就必须使用https.

# 集成本地特征

最好的更新pwa的方式是利用用户操作系统的web功能。

## 快捷方式

比如:![](https://docs.pwabuilder.com/assets/home/native-features/shortcuts.png)

### 实现快捷方式

只需要在manifest中添加可用的`shortcuts`,`shortcuts`字段会接受`shortcut`对象列表，下面是一个例子:
```json
"shortcuts": [
  {
    "name": "News Feed",
    "url": "/feed",
    "description": "Noteworthy news from today."
  },
  {
    "name": "New Post",
    "url": "/post",
    "description": "Create a new post."
  }
]

```
- name,名字
- url, 导航
- description, 描述

还有一些其他的字段:
```json
"shortcuts": [
  {
    "name": "News Feed",
    "short_name": "Feed",
    "url": "/feed",
    "description": "Noteworthy news from today.",
    "icons": [
      {
        "src": "assets/icons/news.png",
        "type": "image/png",
        "purpose": "any"
      }
    ]
  }
]

```
`icons`允许我们在快捷方式中显示自定义的图标。

## 窗口覆盖控制
启用这个特征允许我们使用CSS和js自定义窗口大小，要启用这个特征在manifest中设置`display_override`：
```json
{
    “display_override”: [“window-controls-overlay”],
    "display": "standalone"
}

```

### 自定义CSS
标题栏区域也是可用的，我们可以使用CSS环境变量来使用。下面是一些关键变量:
- `titlebar-area-x`
- `titlebar-area-y`
- `titlebar-area-width`
- `titlebar-area-height`
下面是一个示例:
```css
.titleBar {
    position: fixed;
    left: env(titlebar-area-x, 0);
    top: env(titlebar-area-y, 0);
    width: env(titlebar-area-width, 100%);
    height: env(titlebar-area-height, 40px);
    -webkit-app-region: drag;
    app-region: drag;
}

```
`-webkit-app-region`和`app-region`要设置成`drag`.

## web共享api
注:必须要https
![示例](https://docs.pwabuilder.com/assets/home/native-features/pwinter-share.jpg)

使用web share api可以共享给或者从pwa共享。
### share from pwa
最简单的从pwa共享的方式是web连接，可以使用一个单独的函数实现:
```js
async function shareLink(shareTitle, shareText, link) {
    const shareData = {
        title: shareTitle,
        text: shareText,
        url: link,
    };
    try {
        await navigator.share(shareData);
    } catch (e) {
        console.error(e);
    }
}

```
只需要调用`navigator.share()`,然后传进来想要共享的数据。

也可以使用这个web api来共享文件:
```js
async function shareFiles(filesArray, shareTitle, shareText) {
    if (navigator.canShare && navigator.canShare({ files: filesArray })) {
        try {
            await navigator.share({
                files: filesArray,
                title: shareTitle,
                text: shareText
            });
        } catch (error) {
            console.log('Sharing failed', error);
        }
    } else {
        console.log(`System doesn't support sharing.`);
    }
};

```

唯一多的是确认文件类型。

### share to pwa

可以让pwa从本地操作系统接收共享文件，可以在manifest中添加`share_target`成员:
```json
"share_target": {
      "action": "index.html?share-action",
      "method": "GET",
      "enctype": "application/x-www-form-urlencoded",
      "params": {
        "title": "title",
        "text": "text",
        "url": "url"
      }
    }

```
关键的字段是`action`,这允许我们设置明确的url来打开和处理共享的链接类型。如果想要基于这个共享连接执行某个功能，可以让这个页面解析连接然后决定如何处理共享数据。

## Badging

![示例](https://docs.pwabuilder.com/assets/home/native-features/badging-task-bar.png)

### 显示和清除Badges
```js
if ('setAppBadge' in navigator) {
  navigator.setAppBadge(1);
}

```
上述代码检查`setAppBadge`功能是否可用然后调用`navigator.setAppBadge(1)`显示值1.要进行清空使用函数:
```js
navigator.clearAppBadge();

```
或者调用:
```js
navigator.setAppBadge(0);
```