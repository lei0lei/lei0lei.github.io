---
layout: default
title: pwa-gh
permalink: /web/pwa/pwa-gh
parent: PWA
grand_parent: web
has_toc: true
---

在本页面中，我们将会介绍在github pages中自动部署pwa应用。

参考资源: 
- https://docs.pwabuilder.com/#/starter/quick-start
- https://microsoft.github.io/win-student-devs/#/30DaysOfPWA/core-concepts/
- https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API
- https://create-react-app.dev/docs/making-a-progressive-web-app/
- https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps
- https://github.com/pwa-builder/pwa-starter
- https://web.dev/learn/pwa/

# 简介
PWA应用相比普通的web应用多了:
- web app manifest
- service worker
注意，serviceworker必须使用https.
## web app manifest
这是一个json文件，告诉浏览器web应用是一个PWA，可以安装到系统上。manifest还包含一些app相关的信息比如标题，主题颜色，描述。
也可以启用一些本地功能比如快捷方式和display mode.

## service worker
这是一类[web worker](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Using_web_workers),作为网络的代理，拦截所有PWA的请求。它们活跃在应用之外是事件驱动的单独线程。

service worker赋予了PWA离线的能力，可以缓存一部分资源并在网络存在问题的时候进行一些请求处理。从PWA出来的和到PWA的所有请求都要经过service worker,
有相当多的策略关于如何缓存和获取必要的资源实现离线能力。

## 可安装性
带manifest和service worker的应用可以实现本地安装，这是叫做渐进式的原因。和本地应用一样，pwa可以放在桌面，任务栏等地方。pwa可以从eage或者chrome安装也可以从手机的app store安装。

好的pwa一般会有以下的特点:
- Smooth, modern, and consistent UI and navigation experience (for example, the view shouldn’t refresh on navigation)

- Reliable offline experience. Even if your app is heavily network dependent, it should remain basically responsive when offline.

- Takes advantage (when appropriate) of modern web capabilities to deliver an integrated application experience.
