---
layout: default
title: pytorch lightning
permalink: /python/pytorch-lightning
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

# 7个关键步骤
## 定义LightningModule
LightningModule把pytorch的nn.module放到了一起，数据处理，训练等步骤都包在一个类中。
```py
import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)
```
## 定义数据集
lightning支持任何iterable类型。
```py
dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset)
```

## 模型训练
Lightning 的Trainer 可以混合使用任意的 LightningModule 和任意的数据集。
```py
# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = pl.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```
Trainer支持40多种tricks.
- Epoch and batch iteration
- optimizer.step(), loss.backward(), optimizer.zero_grad() 调用
- 调用model.eval(), 阶段禁用或者启用grads.
- Checkpoint 保存和加载
- Tensorboard (see loggers options)
- Multi-GPU 
- TPU
- 16-bit precision AMP 

## 模型的使用
支持产品级部署
```py
# load checkpoint
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

# choose your trained nn.Module
encoder = autoencoder.encoder
encoder.eval()

# embed 4 fake images!
fake_image_batch = Tensor(4, 28 * 28)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
```

## 训练可视化

```py
tensorboard --logdir .
```
## Supercharge training
在trainer中传入参数支持高级的训练特征。

```py
# train on 4 GPUs
trainer = Trainer(
    devices=4,
    accelerator="gpu",
 )

# train 1TB+ parameter models with Deepspeed/fsdp
trainer = Trainer(
    devices=4,
    accelerator="gpu",
    strategy="deepspeed_stage_2",
    precision=16
 )

# 20+ helpful flags for rapid idea iteration
trainer = Trainer(
    max_epochs=10,
    min_epochs=5,
    overfit_batches=1
 )

# access the latest state of the art techniques
trainer = Trainer(callbacks=[StochasticWeightAveraging(...)])
```
lightning提供了额外的灵活度，可以自定义训练循环，
```py
class LitAutoEncoder(pl.LightningModule):
    def backward(self, loss):
        loss.backward()
```

扩展trainer：

```py
trainer = Trainer(callbacks=[AWSCheckpoints()])
```
# 模型训练

## import
```py
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning.pytorch as pl
```

## 定义nn.Modules
```py
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)
```

## 定义LightningModule
- `training_step`定义`nn.Modules`的交互
- `configure_optimizers`定义模型的优化器
```py
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
```
## 定义训练集
定义pytorch的`DataLoader`
```py
dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
```
## 模型训练
使用`Trainer`训练模型

```py
# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
trainer = pl.Trainer()
trainer.fit(model=autoencoder, train_dataloaders=train_loader)
```
## 干掉训练循环
`Trainer`在背地里为我们做了以下事情:

```py
autoencoder = LitAutoEncoder(Encoder(), Decoder())
optimizer = autoencoder.configure_optimizers()

for batch_idx, batch in enumerate(train_loader):
    loss = autoencoder.training_step(batch, batch_idx)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

# 添加验证集和测试集


为了确保模型在未见过的数据集上也能使用，数据集一般会分成训练集和测试集，测试集在训练阶段不使用。

## 分割数据集

```py
import torch.utils.data as data
from torchvision import datasets
import torchvision.transforms as transforms

# Load data sets
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True, train=False, transform=transform)
```
## 定义测试循环

实现`test_step`方法，

```py
class LitAutoEncoder(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)
```
## 训练之后加入测试步骤

```py
from torch.utils.data import DataLoader

# initialize the Trainer
trainer = Trainer()

# test the model
trainer.test(model, dataloaders=DataLoader(test_set))
```

## 添加验证循环
在训练集中分出一部分作为验证集

```py
# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
```
添加`validation_step`

```py
class LitAutoEncoder(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)
```

```py
from torch.utils.data import DataLoader

train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)

# train with both splits
trainer = Trainer()
trainer.fit(model, train_loader, valid_loader)
```
# 保存模型过程
lightning的checkpoint包含以下内容:
- 16-bit scaling factor (if using 16-bit precision training)
- Current epoch
- Global step
- LightningModule’s state_dict
- State of all optimizers
- State of all learning rate schedulers
- State of all callbacks (for stateful callbacks)
- State of datamodule (for stateful datamodules)
- The hyperparameters (init arguments) with which the model was created
- The hyperparameters (init arguments) with which the datamodule was created
- State of Loops

## 保存ckpt

```py
# saves checkpoints to 'some/path/' at every epoch end
trainer = Trainer(default_root_dir="some/path/")
```

## 加载

```py
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

# disable randomness, dropout, etc...
model.eval()

# predict with the model
y_hat = model(x)
```

超参数保存：

```py
class MyLightningModule(LightningModule):
    def __init__(self, learning_rate, another_parameter, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
```

```py
checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
print(checkpoint["hyper_parameters"])
# {"learning_rate": the_value, "another_parameter": the_other_value}
```

```py
model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
print(model.learning_rate)
```

使用其他参数初始化:

如果初始化`LightningModule`时使用了`self.save_hyperparameters()`，可以使用不同的超参数初始化模型。

```py
# if you train and save the model like this it will use these values when loading
# the weights. But you can overwrite this
LitModel(in_dim=32, out_dim=10)

# uses in_dim=32, out_dim=10
model = LitModel.load_from_checkpoint(PATH)

# uses in_dim=128, out_dim=10
model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)
```

## nn.Module from checkpoint

lightning的ckpt和torch原生的`nn.Modules`完全匹配。

```py
checkpoint = torch.load(CKPT_PATH)
print(checkpoint.keys())
```

假设创建了`LightningModule`

```py
class Encoder(nn.Module):
    ...


class Decoder(nn.Module):
    ...


class Autoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder, *args, **kwargs):
        ...


autoencoder = Autoencoder(Encoder(), Decoder())
```

```py
checkpoint = torch.load(CKPT_PATH)
encoder_weights = checkpoint["encoder"]
decoder_weights = checkpoint["decoder"]
```

## 禁用ckpt

```py
trainer = Trainer(enable_checkpointing=False)
```

## 恢复训练

```py
model = LitModel()
trainer = Trainer()

# automatically restores model, epoch, step, LR schedulers, etc...
trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
```

# 提前终止训练

重写`on_train_batch_start()`来提前终止训练。

`EarlyStopping` 回调函数可以监控一个metric并在模型训练没有提升的时候提前终止，启用这个功能使用以下过程:


- Import EarlyStopping callback.

- Log the metric you want to monitor using log() method.

- Init the callback, and set monitor to the logged metric of your choice.

- Set the mode based on the metric needs to be monitored.

- Pass the EarlyStopping callback to the Trainer callbacks flag.


```py
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


class LitModel(LightningModule):
    def validation_step(self, batch, batch_idx):
        loss = ...
        self.log("val_loss", loss)


model = LitModel()
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model)
```

可以自定义callback的行为:

```py
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
trainer = Trainer(callbacks=[early_stop_callback])
```

一些其他的参数:

- stopping_threshold: Stops training immediately once the monitored quantity reaches this threshold. It is useful when we know that going beyond a certain optimal value does not further benefit us.

- divergence_threshold: Stops training as soon as the monitored quantity becomes worse than this threshold. When reaching a value this bad, we believes the model cannot recover anymore and it is better to stop early and run with different initial conditions.

- check_finite: When turned on, it stops training if the monitored metric becomes NaN or infinite.

- check_on_train_epoch_end: When turned on, it checks the metric at the end of a training epoch. Use this only when you are monitoring any metric logged within training-specific hooks on epoch-level.

```py
class MyEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)
```

{: .note :}
The `EarlyStopping` callback runs at the end of every validation epoch by default. However, the frequency of validation can be modified by setting various parameters in the `Trainer`, for example `check_val_every_n_epoch` and `val_check_interval`. It must be noted that the patience parameter counts the number of validation checks with no improvement, and not the number of training epochs. Therefore, with parameters `check_val_every_n_epoch=10` and `patience=3`, the trainer will perform at least 40 training epochs before being stopped.

# 迁移学习

## 使用预训练的`LightningModule`

```py
class Encoder(torch.nn.Module):
    ...


class AutoEncoder(LightningModule):
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()


class CIFAR10Classifier(LightningModule):
    def __init__(self):
        # init the pretrained LightningModule
        self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH)
        self.feature_extractor.freeze()

        # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        ...
```

```py
import torchvision.models as models


class ImagenetTransferLearning(LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        ...
```

```py
model = ImagenetTransferLearning()
trainer = Trainer()
trainer.fit(model)
```

```py
model = ImagenetTransferLearning.load_from_checkpoint(PATH)
model.freeze()

x = some_images_from_cifar10()
predictions = model(x)
```

## Bert

```py
class BertMNLIFinetuner(LightningModule):
    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased", output_attentions=True)
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):
        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn
```

# 命令行中配置超参数

## ArgumentParser

```py
from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--devices", type=int, default=2)

# Hyperparameters for the model
parser.add_argument("--layer_1_dim", type=int, default=128)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

# Use the parsed arguments in your program
trainer = Trainer(devices=args.devices)
model = MyModel(layer_1_dim=args.layer_1_dim)
```

```sh
python trainer.py --layer_1_dim 64 --devices 1
```
## [lightning cli](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)

# debug,可视化以及寻找瓶颈

## debug
### 设置断点

```py
def function_to_debug():
    x = 2

    # set breakpoint
    import pdb

    pdb.set_trace()
    y = x**2
```

### 跑一遍代码

`fast_dev_run`会跑5个batch的训练验证和预测

```py
Trainer(fast_dev_run=True)
```

```py
Trainer(fast_dev_run=7)
```
{: .note :}
This argument will disable tuner, checkpoint callbacks, early stopping callbacks, loggers and logger callbacks like `LearningRateMonitor` and `DeviceStatsMonitor`.

### 缩短epoch长度
比如使用20%数据集作为训练，1%数据集作为验证

```py
# use only 10% of training data and 1% of val data
trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)

# use 10 batches of train and 5 batches of val
trainer = Trainer(limit_train_batches=10, limit_val_batches=5)
```

### 简单检查
在训练开始的时候进行两步验证。

```py
trainer = Trainer(num_sanity_val_steps=2)
```

### 显示`LightningModule`权重summary

```py
trainer.fit(...)
```

要给子模块添加summary,使用:

```py
from lightning.pytorch.callbacks import ModelSummary

trainer = Trainer(callbacks=[ModelSummary(max_depth=-1)])
```

不调用`.fit`的情况下打印summary:

```py
from lightning.pytorch.utilities.model_summary import ModelSummary

model = LitModel()
summary = ModelSummary(model, max_depth=-1)
print(summary)
```

关闭功能使用:

```py
Trainer(enable_model_summary=False)
```

### 查找代码瓶颈

```py
trainer = Trainer(profiler="simple")
```

要查看每个函数的运行时间，使用：

```py
trainer = Trainer(profiler="advanced")
```

输出到文件中:

```py
from lightning.pytorch.profilers import AdvancedProfiler

profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
trainer = Trainer(profiler=profiler)
```

查看加速器效果:

```py
from lightning.pytorch.callbacks import DeviceStatsMonitor

trainer = Trainer(callbacks=[DeviceStatsMonitor()])
```

CPU metrics will be tracked by default on the CPU accelerator. To enable it for other accelerators set `DeviceStatsMonitor(cpu_stats=True)`. To disable logging CPU metrics, you can specify `DeviceStatsMonitor(cpu_stats=False)`.

### 实验跟踪和可视化

#### metrics跟踪

```py
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        value = ...
        self.log("some_value", value)
```
使用`self.log`方法.

记录多个metrics,使用:

```py
values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
self.log_dict(values)
```

要在进度条里查看使用，

```py
self.log(..., prog_bar=True)
```

浏览器中查看，略

metric积累，略

目录保存：

```py
Trainer(default_root_dir="/your/custom/path")
```

# 模型推理
## 产品级部署-1
### 加载ckpt并预测

```py
model = LitModel.load_from_checkpoint("best_model.ckpt")
model.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)
```

### `LightningModule`添加预测过程

```py
class MyModel(LightningModule):
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
```

把dataloader加载到trainer

```py
data_loader = DataLoader(...)
model = MyModel()
trainer = Trainer()
predictions = trainer.predict(model, data_loader)
```

### 添加复杂的推理逻辑

```py
class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iteration):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iteration

    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        return pred
```

### 使用分布式推理

```py
import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        torch.save(batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt"))


# or you can set `writer_interval="batch"` and override `write_on_batch_end` to save
# predictions at batch level
pred_writer = CustomWriter(output_dir="pred_path", write_interval="epoch")
trainer = Trainer(accelerator="gpu", strategy="ddp", devices=8, callbacks=[pred_writer])
model = BoringModel()
trainer.predict(model, return_predictions=False)
```

## 产品级部署-2

### 使用pytorch

```py
import torch


class MyModel(nn.Module):
    ...


model = MyModel()
checkpoint = torch.load("path/to/lightning/checkpoint.ckpt")
model.load_state_dict(checkpoint["state_dict"])
model.eval()
```

### 从lightning中提取`nn.Modules`

```py
class Encoder(nn.Module):
    ...


class Decoder(nn.Module):
    ...


class AutoEncoderProd(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.encoder(x)


class AutoEncoderSystem(LightningModule):
    def __init__(self):
        super().__init__()
        self.auto_encoder = AutoEncoderProd()

    def forward(self, x):
        return self.auto_encoder.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.auto_encoder.encoder(x)
        y_hat = self.auto_encoder.decoder(y_hat)
        loss = ...
        return loss


# train it
trainer = Trainer(devices=2, accelerator="gpu", strategy="ddp")
model = AutoEncoderSystem()
trainer.fit(model, train_dataloader, val_dataloader)
trainer.save_checkpoint("best_model.ckpt")


# create the PyTorch model and load the checkpoint weights
model = AutoEncoderProd()
checkpoint = torch.load("best_model.ckpt")
hyper_parameters = checkpoint["hyper_parameters"]

# if you want to restore any hyperparameters, you can pass them too
model = AutoEncoderProd(**hyper_parameters)

model_weights = checkpoint["state_dict"]

# update keys by dropping `auto_encoder.`
for key in list(model_weights):
    model_weights[key.replace("auto_encoder.", "")] = model_weights.pop(key)

model.load_state_dict(model_weights)
model.eval()
x = torch.randn(1, 64)

with torch.no_grad():
    y_hat = model(x)
```

# GPU训练

## 代码修改

```py
# before lightning
def forward(self, x):
    x = x.cuda(0)
    layer_1.cuda(0)
    x_hat = layer_1(x)


# after lightning
def forward(self, x):
    x_hat = layer_1(x)
```

使用`tensor.to`和`register_buffer`

```py
# before lightning
def forward(self, x):
    z = torch.Tensor(2, 3)
    z = z.cuda(0)


# with lightning
def forward(self, x):
    z = torch.Tensor(2, 3)
    z = z.to(x)
```

`LightningModule`知道自己处在哪个设备上，使用`self.device`. 有时需要把tensor存储为模块属性。但是如果它们不是参数仍然会存在cpu上，将这个tensor注册为buffer使用`register_buffer()`.

```py
class LitModel(LightningModule):
    def __init__(self):
        ...
        self.register_buffer("sigma", torch.eye(3))
        # you can now access self.sigma anywhere in your module
```

### Remove samplers

sampler是自动处理的。

### 同步

在分布式模式下必须保证验证和测试step的logging调用在进程间同步，可以给`self.log`添加`sync_dist=True`，这在下游的任务比如测试最好的ckpt比较重要。
如果使用内建的metric或者使用`TorchMetrics`自定义metric会进行自动的处理更新。

```py
def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
    self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)


def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
    self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
```

It is possible to perform some computation manually and log the reduced result on rank 0 as follows:

```py
def __init__(self):
    super().__init__()
    self.outputs = []


def test_step(self, batch, batch_idx):
    x, y = batch
    tensors = self(x)
    self.outputs.append(tensors)
    return tensors


def on_test_epoch_end(self):
    mean = torch.mean(self.all_gather(self.outputs))
    self.outputs.clear()  # free memory

    # When logging only on rank 0, don't forget to add
    # `rank_zero_only=True` to avoid deadlocks on synchronization.
    # caveat: monitoring this is unimplemented. see https://github.com/Lightning-AI/lightning/issues/15852
    if self.trainer.is_global_zero:
        self.log("my_reduced_metric", mean, rank_zero_only=True)
```

### pickleable model
在并行模式下可能出现以下错误:

```
self._launch(process_obj)
File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/popen_spawn_posix.py", line 47,
in _launch reduction.dump(process_obj, fp)
File "/net/software/local/python/3.6.5/lib/python3.6/multiprocessing/reduction.py", line 60, in dump
ForkingPickler(file, protocol).dump(obj)
_pickle.PicklingError: Can't pickle <function <lambda> at 0x2b599e088ae8>:
attribute lookup <lambda> on __main__ failed
```

这表明并行模式下模型，优化器,dataloader...中存在无法保存的东西，这是由pytorch限制的。

## gpu训练

默认情况下会尽可能在gpu上进行训练:

```py
# run on as many GPUs as available by default
trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
# equivalent to
trainer = Trainer()

# run on one GPU
trainer = Trainer(accelerator="gpu", devices=1)
# run on multiple GPUs
trainer = Trainer(accelerator="gpu", devices=8)
# choose the number of devices automatically
trainer = Trainer(accelerator="gpu", devices="auto")
```

{: .note :}
Setting accelerator="gpu" will also automatically choose the “mps” device on Apple sillicon GPUs. If you want to avoid this, you can set accelerator="cuda" instead.

可以选择gpu设备

```py
# DEFAULT (int) specifies how many GPUs to use per node
Trainer(accelerator="gpu", devices=k)

# Above is equivalent to
Trainer(accelerator="gpu", devices=list(range(k)))

# Specify which GPUs to use (don't use when running on cluster)
Trainer(accelerator="gpu", devices=[0, 1])

# Equivalent using a string
Trainer(accelerator="gpu", devices="0, 1")

# To use all available GPUs put -1 or '-1'
# equivalent to list(range(torch.cuda.device_count()))
Trainer(accelerator="gpu", devices=-1)
```

检测可以使用的gpu设备:

```py
from lightning.pytorch.accelerators import find_usable_cuda_devices

# Find two GPUs on the system that are not already occupied
trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2))

from lightning.fabric.accelerators import find_usable_cuda_devices

# Works with Fabric too
fabric = Fabric(accelerator="cuda", devices=find_usable_cuda_devices(2))
```

当gpu被设置为`exclusive compute mode`时比较有用。

# 项目模块化

## datamodule

<iframe width="420" height="315" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pt_dm_vid.m4v" frameborder="0"></iframe>

datamodule是用来处理数据的类。下面是5个步骤:

1. Download / tokenize / process.
2. Clean and (maybe) save to disk.
3. Load inside Dataset.
4. Apply transforms (rotate, tokenize, etc…).
5. Wrap inside a DataLoader.
然后可以使用:

```py
model = LitClassifier()
trainer = Trainer()

imagenet = ImagenetDataModule()
trainer.fit(model, datamodule=imagenet)

cifar10 = CIFAR10DataModule()
trainer.fit(model, datamodule=cifar10)
```

datamodule解决了以下几个问题:

- what splits did you use?

- what transforms did you use?

- what normalization did you use?

- how did you prepare/tokenize the data?

在pytorch中需要这样写:

```py
# regular PyTorch
test_data = MNIST(my_path, train=False, download=True)
predict_data = MNIST(my_path, train=False, download=True)
train_data = MNIST(my_path, train=True, download=True)
train_data, val_data = random_split(train_data, [55000, 5000])

train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
predict_loader = DataLoader(predict_data, batch_size=32)
```

等效的在lightning中:

```py
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_test = MNIST(self.data_dir, train=False)
        self.mnist_predict = MNIST(self.data_dir, train=False)
        mnist_full = MNIST(self.data_dir, train=True)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...
```

然后就可以复用:

```py
mnist = MNISTDataModule(my_path)
model = LitClassifier()

trainer = Trainer()
trainer.fit(model, mnist)
```

下面是一个更复杂的例子:

```py
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
```

要定义datamodule需要实现以下方法:

- prepare_data (how to download, tokenize, etc…)
- setup (how to split, define dataset, etc…)
- train_dataloader
- val_dataloader
- test_dataloader
- predict_dataloader

### prepare_data

用多个进程下载和保存数据可能导致数据冲突，Lightning可以确保`prepare_data()`只在cpu的一个进程上调用。对于多节点训练，这个hook取决于`prepare_data_per_node`。`setup()`会在`prepare_data`之后进行调用，there is a barrier in between which ensures that all the processes proceed to setup once the data is prepared and available for use.

- download, i.e. download data only once on the disk from a single process

- tokenize. Since it’s a one time process, it is not recommended to do it on all processes

```py
class MNISTDataModule(pl.LightningDataModule):
    def prepare_data(self):
        # download
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())
```

{: .warning :}
`prepare_data` is called from the main process. It is not recommended to assign state here (e.g. self.x = y) since it is called on a single process and if you assign states here then they won’t be available for other processes.

### setup
有时想要在每块GPU上进行数据操作，使用`setup()`:

- count number of classes

- build vocabulary

- perform train/val/test splits

- create datasets

- apply transforms (defined explicitly in your datamodule)

```py
import lightning.pytorch as pl


class MNISTDataModule(pl.LightningDataModule):
    def setup(self, stage: str):
        # Assign Train/val split(s) for use in Dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign Test split(s) for use in Dataloaders
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=self.transform)
```
对于NLP可能想要获得文本tooken,可以:

```py
class LitDataModule(LightningDataModule):
    def prepare_data(self):
        dataset = load_Dataset(...)
        train_dataset = ...
        val_dataset = ...
        # tokenize
        # save it to disk

    def setup(self, stage):
        # load it back here
        dataset = load_dataset_from_disk(...)
```

`stage`参数用来为trainer设置，trainer.{fit,validate,test,predict}.

{: .note :}
> setup is called from every process across all the nodes. Setting state here is recommended.
> 
> teardown can be used to clean up the state. It is also called from every process across all the nodes.

### train_dataloader

`train_dataloader()`方法用来生成训练dataloader.通常只是封装在`setup`中封装的dataset. trainer的`fit()`方法将会使用这个dataloader. 

```py
import lightning.pytorch as pl

class MNISTDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)
```

### val_dataloader

### test_dataloader

### predict_dataloader


### transfer_batch_to_device


### on_before_batch_transfer


### on_after_batch_transfer


### load_state_dict


### state_dict


### teardown


### prepare_data_per_node


### 使用datamodule

datamodule的使用非常简单:

```py
dm = MNISTDataModule()
model = Model()
trainer.fit(model, datamodule=dm)
trainer.test(datamodule=dm)
trainer.validate(datamodule=dm)
trainer.predict(datamodule=dm)
```

如果需要数据集的某些信息才能构建模型，手动运行`prepare_data`和`setup`:

```py
dm = MNISTDataModule()
dm.prepare_data()
dm.setup(stage="fit")

model = Model(num_classes=dm.num_classes, width=dm.width, vocab=dm.vocab)
trainer.fit(model, dm)

dm.setup(stage="test")
trainer.test(datamodule=dm)
```

You can access the current used datamodule of a trainer via `trainer.datamodule` and the current used dataloaders via the trainer properties `train_dataloader()`, `val_dataloaders()`, `test_dataloaders()`, and `predict_dataloaders()`.

### 在pytorch中使用DataModules

```py
# download, etc...
dm = MNISTDataModule()
dm.prepare_data()

# splits/transforms
dm.setup(stage="fit")

# use data
for batch in dm.train_dataloader():
    ...

for batch in dm.val_dataloader():
    ...

dm.teardown(stage="fit")

# lazy load test data
dm.setup(stage="test")
for batch in dm.test_dataloader():
    ...

dm.teardown(stage="test")
```

### datamodule中的超参数

```py
import lightning.pytorch as pl

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def configure_optimizers(self):
        # access the saved hyperparameters
        opt = optim.Adam(self.parameters(), lr=self.hparams.lr)
```

### 保存datamodule state

```py
class LitDataModule(pl.DataModuler):
    def state_dict(self):
        # track whatever you want here
        state = {"current_train_batch_index": self.current_train_batch_index}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.current_train_batch_index = state_dict["current_train_batch_index"]
```

## CLI中配置超参数-1

`LightningCLI`用来减轻CLI实现难度，要使用这个类，需要额外的lightning功能，

```sh
pip install "pytorch-lightning[extra]"
```

### 实现CLI
实例化一个`LightningCLI`对象，然后给`LightningModule`参数，也可以多给一个`LightningDataModule`参数。

```py
# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
```
现在模型可以通过CLI管理:

```sh
python main.py --help
```
会输出:

```
usage: main.py [-h] [-c CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
        {fit,validate,test,predict,tune} ...

pytorch-lightning trainer command line tool

optional arguments:
-h, --help            Show this help message and exit.
-c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
--print_config [={comments,skip_null,skip_default}+]
                        Print configuration and exit.

subcommands:
For more details of each subcommand add it as argument followed by --help.

{fit,validate,test,predict,tune}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set.
    predict             Run inference on your data.
    tune                Runs routines to tune hyperparameters before training.
```

### 使用CLI训练模型

```sh
python main.py fit
```
`--help`参数查看可用选项:

```
$ python main.py fit --help

usage: main.py [options] fit [-h] [-c CONFIG]
                            [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
                            ...
                            [--ckpt_path CKPT_PATH]
    --trainer.logger LOGGER

optional arguments:
<class '__main__.DemoModel'>:
    --model.out_dim OUT_DIM
                            (type: int, default: 10)
    --model.learning_rate LEARNING_RATE
                            (type: float, default: 0.02)
<class 'lightning.pytorch.demos.boring_classes.BoringDataModule'>:
--data CONFIG         Path to a configuration file.
--data.data_dir DATA_DIR
                        (type: str, default: ./)
```

改变参数:

```sh
# change the learning_rate
python main.py fit --model.learning_rate 0.1

# change the output dimensions also
python main.py fit --model.out_dim 10 --model.learning_rate 0.1

# change trainer and data arguments too
python main.py fit --model.out_dim 2 --model.learning_rate 0.1 --data.data_dir '~/' --trainer.logger False
```
{: .note :}
 `LightningModule` 和 `LightningDataModule`类中的`__init__`的参数在CLI中发挥作用，因此，想要一个参数可以配置，将其添加到类的`__init__`中。 最好在docstring中描述这些参数，这样可以通过`--help`进行查看，最好加上type hint.

## CLI中配置超参数-2
lightning支持混合使用模型和数据集，比如:

```sh
# Mix and match anything
$ python main.py fit --model=GAN --data=MNIST
$ python main.py fit --model=Transformer --data=MNIST
```

`LightningCLI`可以方便实现这一功能，不用像下面一样写过多代码:

```py
# choose model
if args.model == "gan":
    model = GAN(args.feat_dim)
elif args.model == "transformer":
    model = Transformer(args.feat_dim)
...

# choose datamodule
if args.data == "MNIST":
    datamodule = MNIST()
elif args.data == "imagenet":
    datamodule = Imagenet()
...

# mix them!
trainer.fit(model, datamodule)
```

### 多个LightningModules

```py
# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class Model1(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()


cli = LightningCLI(datamodule_class=BoringDataModule)
```

现在可以在CLI中选择模型:

```sh
# use Model1
python main.py fit --model Model1

# use Model2
python main.py fit --model Model2
```

{: .note :}
如果不使用`model_class`参数，可以使用基类以及`subclass_mode_model=True`，这样cli只能接收给定基类的子类模型。

### 多个 LightningDataModules

在`LightningCLI`中使用`datamodule_class`参数：

```py
# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class FakeDataset1(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset1", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


class FakeDataset2(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset2", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


cli = LightningCLI(DemoModel)
```

现在可以使用任意数据集:

```sh
# use Model1
python main.py fit --data FakeDataset1

# use Model2
python main.py fit --data FakeDataset2
```

{: .note :}
Instead of omitting the `datamodule_class` parameter, you can give a base class and `subclass_mode_data=True`. This will make the CLI only accept data modules that are a subclass of the given base class.



### 多个优化器

使用标准的优化器:

```sh
python main.py fit --optimizer AdamW

python main.py fit --optimizer SGD --optimizer.lr=0.01
```

自定义优化器:

```py
# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class LitAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using LitAdam", "⚡")
        super().step(closure)


class FancyAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using FancyAdam", "⚡")
        super().step(closure)


cli = LightningCLI(DemoModel, BoringDataModule)
```

```sh
# use LitAdam
python main.py fit --optimizer LitAdam

# use FancyAdam
python main.py fit --optimizer FancyAdam
```

### 多个scheduler

```sh
python main.py fit --lr_scheduler CosineAnnealingLR
python main.py fit --lr_scheduler=ReduceLROnPlateau --lr_scheduler.monitor=epoch
```

```py
# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class LitLRScheduler(torch.optim.lr_scheduler.CosineAnnealingLR):
    def step(self):
        print("⚡", "using LitLRScheduler", "⚡")
        super().step()


cli = LightningCLI(DemoModel, BoringDataModule)
```

```py
# LitLRScheduler
python main.py fit --lr_scheduler LitLRScheduler
```


### 其他包中的类

```py
from lightning.pytorch.cli import LightningCLI
import my_code.models  # noqa: F401
import my_code.data_modules  # noqa: F401
import my_code.optimizers  # noqa: F401

cli = LightningCLI()
```

```sh
python main.py fit --model Model1 --data FakeDataset1 --optimizer LitAdam --lr_scheduler LitLRScheduler
```

{: .note :}
The `# noqa: F401` comment avoids a linter warning that the import is unused.

```sh
python main.py fit --model my_code.models.Model1
```

### 模型help
用多个模型或数据集时CLI的help不会包含对应的参数，用该用以下的方式:

```sh
python main.py fit --model.help Model1
python main.py fit --data.help FakeDataset2
python main.py fit --optimizer.help Adagrad
python main.py fit --lr_scheduler.help StepLR
```

## CLI中配置超参数-3

随着参数的增多从CLI中引入参数变得不现实，LightningCLI可以支持从配置文件中接收输入.

```sh
python main.py fit --config config.yaml
```
默认LightningCLI自动保存完整的YAML配置在log目录下。

自动保存通过特定的回调`SaveConfigCallback`实现，这个回调时自动添加到Trainer上的，要禁用，实例化`LightningCLI`时传入`save_config_callback=None`

要改变名字使用:

```py
cli = LightningCLI(..., save_config_kwargs={"config_filename": "name.yaml"})
```

### 为CLI准备config文件
不运行命令只打印参数:

```sh
python main.py fit --print_config
```
会生成:

```
seed_everything: null
trainer:
  logger: true
  ...
model:
  out_dim: 10
  learning_rate: 0.02
data:
  data_dir: ./
ckpt_path: null
```

```sh
python main.py fit --model DemoModel --print_config
```
生成:

```
seed_everything: null
trainer:
  ...
model:
  class_path: lightning.pytorch.demos.boring_classes.DemoModel
  init_args:
    out_dim: 10
    learning_rate: 0.02
ckpt_path: null
```
{: .note :}
> 标准的实验过程是:
> ```sh
> # Print a configuration to have as reference
> python main.py fit --print_config > config.yaml
> # Modify the config to your liking - you can remove all default arguments
> nano config.yaml
> # Fit your model using the edited configuration
> python main.py fit --config config.yaml
> ```

如果模型定义为:

```py
# model.py
class MyModel(pl.LightningModule):
    def __init__(self, criterion: torch.nn.Module):
        self.criterion = criterion
```

config将会是:

```yaml
model:
  class_path: model.MyModel
  init_args:
    criterion:
      class_path: torch.nn.CrossEntropyLoss
      init_args:
        reduction: mean
    ...
```

{: .note :}
Lighting automatically registers all subclasses of `LightningModule`, so the complete import path is not required for them and can be replaced by the class name.

### 组合配置文件
可以使用多个配置文件:

```
# config_1.yaml
trainer:
  num_epochs: 10
  ...

# config_2.yaml
trainer:
  num_epochs: 20
  ...
```
会使用最后一个配置的值:

```sh
$ python main.py fit --config config_1.yaml --config config_2.yaml
```
一组选项也可以放在多个文件中:

```
# trainer.yaml
num_epochs: 10

# model.yaml
out_dim: 7

# data.yaml
data_dir: ./data
```

```sh
$ python main.py fit --trainer trainer.yaml --model model.yaml --data data.yaml [...]
```

## CLI中配置超参数-4

要自定义子命令的参数，在子命令前传递参数:

```sh
$ python main.py [before] [subcommand] [after]
$ python main.py  ...         fit       ...
```

比如:

```
# config.yaml
fit:
    trainer:
        max_steps: 100
test:
    trainer:
        max_epochs: 10
```

```sh
# full routine with max_steps = 100
$ python main.py --config config.yaml fit

# test only with max_epochs = 10
$ python main.py --config config.yaml test
```
通过环境变量使用config:

```sh
$ python main.py fit --trainer "$TRAINER_CONFIG" --model "$MODEL_CONFIG" [...]
```

直接从环境变量运行:

```py
cli = LightningCLI(..., parser_kwargs={"default_env": True})
```

运行：

```sh
$ python main.py fit --help
```

```
usage: main.py [options] fit [-h] [-c CONFIG]
                            ...

optional arguments:
...
ARG:   --model.out_dim OUT_DIM
ENV:   PL_FIT__MODEL__OUT_DIM
                        (type: int, default: 10)
ARG:   --model.learning_rate LEARNING_RATE
ENV:   PL_FIT__MODEL__LEARNING_RATE
                        (type: float, default: 0.02)
```
现在通过环境变量定义:

```sh
# set the options via env vars
$ export PL_FIT__MODEL__LEARNING_RATE=0.01
$ export PL_FIT__MODEL__OUT_DIM=5

$ python main.py fit
```

设置默认的config文件:

```py
cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"default_config_files": ["my_cli_defaults.yaml"]})
```
或者:

```py
cli = LightningCLI(MyModel, MyDataModule, parser_kwargs={"fit": {"default_config_files": ["my_fit_defaults.yaml"]}})
```

### 变量插入
受限安装

```sh
pip install omegaconf
```

```yaml
model:
  encoder_layers: 12
  decoder_layers:
  - ${model.encoder_layers}
  - 4
```

```py
cli = LightningCLI(MyModel, parser_kwargs={"parser_mode": "omegaconf"})
```

```sh
python main.py --model.encoder_layers=12
```

{: .note :}
变量插入有时并不是正确的方法。当一个参数必须从其他设置得到时，不应该由CLI用户在配置文件中设置，比如data和model需要batch_size相同，那么应该使用参数连接而不是变量插入。

## CLI中配置超参数-5

## CLI中配置超参数-6


# checkpoint


# 实验管理


