---
layout: default
title: mlflow
permalink: /python/mlflow
parent: pytorch
grand_parent: MachineLearning
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

# Quick start
## 安装

```sh
pip install mlflow[extras]
```
## 使用tracking api

Tracking API 可以让我们在代码中设置记录metrics和artifacts并查看运行历史。下面是一个例子:

```py
import os
from random import random, randint
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", randint(0, 100))

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", random())
    log_metric("foo", random() + 1)
    log_metric("foo", random() + 2)

    # Log an artifact (output file)
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("hello world!")
    log_artifacts("outputs")

```

tracking api 会把数据写入./mlrun目录下的文件中，可以运行:

```sh
mlflow ui
```

打开localhost:5000.

## 运行mlflow项目

mlflow允许将代码和依赖打包成项目，没个项目包含本身的代码和MLproject文件(定义了依赖，项目可运行的命令和参数)。

```sh
mlflow run sklearn_elasticnet_wine -P alpha=0.5

mlflow run https://github.com/mlflow/mlflow-example.git -P alpha=5.0
```

## Saving and Serving Models












