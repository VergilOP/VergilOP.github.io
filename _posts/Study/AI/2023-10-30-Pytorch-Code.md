---
layout: post
title: Pytorch Code
author: Ver
date: 2023-10-31 00:00 +0800
categories: [Study, AI]
tags: [machine learning, deep learning]
mermaid: true
math: true
pin: false
---

## 张量(Tensors)

导入torch库

```py
    import torch 
    import numpy as np
```

建立一个随机3x4矩阵

```py
    x = torch.rand(3,4)
    print(x)
    '''
    tensor([[0.7247, 0.7318, 0.3172, 0.6321],
        [0.3736, 0.8269, 0.3117, 0.0403],
        [0.5608, 0.4408, 0.9847, 0.3674]])
    '''
```

直接通过数据建立一个张量

```py
    x = torch.tensor([3, 4.5])
    print(x)
```

通过`size`获得张量的大小

```py
    print(x.size())
    # Besides .size(), one may find .shape works as well
    print(x.shape)
```

张量之间的操作

```py
    x = torch.rand(3,4)
    y = torch.rand(3,4)
    print(x+y)
```

```py
    print(torch.add(x, y))
```

```py
    result = torch.empty(3,4)
    # 直接把结果赋值
    torch.add(x, y, out=result)
    print(result)
```

也可以用`numpy()`b把Torch Tensor转换到Numpy Array

```py
    x = torch.ones(5)
    print(x)
    print(x.dtype)

    y = x.numpy()
    print(y)
    print(y.dtype)
```

torch.from_numpy()函数可以将Numpy数组转换为Torch张量

> 注意n-d array和tensor公用一个底层数据

```py
    x = np.ones(3)
    y = torch.from_numpy(x)

    np.add(x, 1, out=x)
    print(x)
    print(y)
```

```py
    #Use .copy() to only copy the values, and avoid using the same underlying sturcture .
    x = np.ones(3)
    y = torch.from_numpy(x.copy())

    np.add(x, 1, out=x)
    print(x)
    print(y)
```

可以使用`.to`方法将张量移动到任何设备上

```py
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # check whether a GPU is available
    y = torch.ones(3,4, device=device)  # directly create a tensor on GPU
    x = torch.rand(3,4).to(device)      # or just use .to(device) 
    z = x+y
    print(z)
    print(z.device)
```

## 自动梯度(Autograd)

Autograd是PyTorch的核心包，用于自动微分。它使用基于磁带的系统：在前向传播中，它会记录所有对张量的操作；在反向传播中，它重播这些操作来计算梯度。如果一个张量的`.requires_grad`属性设置为`True`，PyTorch会开始跟踪对它的所有操作。完成计算后，调用`.backward()`方法可以自动计算并累积梯度到`.grad`属性中。

```py
    x = torch.ones(2, 2, requires_grad=True)
    print(x)

    # tensor([[1., 1.], [1., 1.]], requires_grad=True)
```

```py
    y = x + 2
    print(y)

    # tensor([[3., 3.], [3., 3.]], grad_fn=<AddBackward0>)
```

Autograd的实现中，`Function`是PyTorch中的一个重要类。`Tensor`和`Function`相互连接，构建一个非循环图，这个图编码了完整的计算历史。每个张量都有一个`.grad_fn`属性，该属性引用了创建该`Tensor`的`Function`。但是，用户直接创建的张量是一个例外，它们的`grad_fn`属性为`None`。当一个张量是由某个操作产生的结果时，如上文中的`y`，它将有一个`grad_fn`。你可以直接打印这个属性的值来查看它。

```py
    print(y.grad_fn)
    print(x.grad_fn)

    # <AddBackward0 object at 0x000001CCA9EAD5D0>
    # None
```
