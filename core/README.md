# ensemble
- [ ] cls0
- [x] cls1 
{
    "dim": 32,
    "lr": 0.1,
    "bs": 32,
    "eps": 512,
    "attack": true,
    "use_ema": false
    "score":90.6,
}
- [x] cls2{
    "dim": 64,
    "lr": 0.05,
    "bs": 32,
    "eps": 2048,
    "attack": false,
    "use_ema": false,
    "score":91.43,
}
- [ ] cls3




# 步骤
确定8个任务的最合适的网络结构

1.提交测试40的cls0
深度网络有没有可能
cls1: emlp
{
    "dim": 32,
    "lr": 0.1,
    "batch_size": 8,
    "epochs": 2048,
    "attack": false,
    "use_ema": false
}
- [x] cls2
{
    "dim": 64,
    "lr": 0.05,
    "batch_size": 8,
    "epochs": 512,
    "attack": true,
    "use_ema": true
} 

- [x] cls3
{
    "dim": 32,
    "lr": 0.05,
    "batch_size": 8,
    "epochs": 1024,
    "attack": false,
    "use_ema": true
}
96.3


- [x] cls4
{
    "dim": 128,
    "lr": 0.01,
    "batch_size": 8,
    "epochs": 2048,
    "attack": false,
    "use_ema": true
}91.3




- [x] cls5
{
    "dim": 512,
    "lr": 0.1,
    "batch_size": 8,
    "epochs": 2048,
    "attack": true,
    "use_ema": false
}0.683


- [x] cls6 {
    "dim": 64,
    "lr": 0.01,
    "batch_size": 8,
    "epochs": 1024,
    "attack": false,
    "use_ema": false
}0.934

- [x] cls7{
    "dim": 128,
    "lr": 0.1,
    "batch_size": 8,
    "epochs": 2048,
    "attack": false,
    "use_ema": false
}0.836

1.诶个微调
2.trick:
* label扰动+-1
* embedding dropout + 大dim
* ema
* attack
* 模型集成投票
3.co-train



## cls0
|model|kdl|kdl(A)|cfg|
|----|----|-----|----|
|simlp|40|-|dim32-bs8-eps2048|
|xdeepfm|38.99|-|dim32-bs8-eps1024|
|crmlp|38.71|-|dim32-bs8-eps1024|
|emlp|35.84|-|dim32-bs8-eps1024|
|trmlp|30.83|-|-|

## cls1
|model|kdl|kdl(A)|cfg|
|----|----|-----|----|
|emlp|90.38|-|dim32-bs8-eps1024|
|crmlp|89.99|-|dim32-bs8-eps1024|
|simlp|82.7|-|dim64-bs8-eps2048|
|xdeepfm|81.49|-|dim32-bs8-eps1024|
|trmlp|75.39|-|-|

## cls2








