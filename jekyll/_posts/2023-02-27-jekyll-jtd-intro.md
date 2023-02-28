---
layout: post
title: how to use jekyll just-the-doc theme
category: howtos
date: 2023-02-27
---




# 导航
## 主页面导航

J-T-D站点的主导航在页面的左侧，如果屏幕的小的话在顶部。主导航可以形成多层的系统，
默认情况下所有的页面都会出现在顶层，除非定义了父页面。

## 有序页面

要明确页面顺序可以在`front matter`中使用`mnav_order`参数。
```yaml
---
layout: default
title: Customization
nav_order: 4
---
```

参数值决定了顶层页面的次序以及相同父页面的子页面。可以为不同父页面的子页面复用这
些参数值。

参数值可以是数字或者字符串，数值型的参数在字符串型的前面，在省略`nav_order`时默
认是页面的title.如果想要页面的顺序独立于页面的title,需要在所有页面上明确设定
`nav_order`,设置了`nav_order`的所有页面出现在以title排序的页面前面。

默认情况下，大写字符在小写字符前面，可以在配置文件中使用`nac_sort:
case_insensitive`来忽略大小写。字符串可以使用引号闭合，数值类型如果使用引号将会
按照文本排序。

## 排除页面


## 子页面


## 附加链接


## 外部导航链接


## TOC 页面导航
