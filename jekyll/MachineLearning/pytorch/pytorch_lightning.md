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










