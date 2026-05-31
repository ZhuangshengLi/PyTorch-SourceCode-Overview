# 自动混合精度

## 背景

早期的主流 GPU 运算精度都是 FP32 的，到了 Volta(V100) 架构，出现了 `Tensor Core`，这是一个更擅长 FP16 矩阵乘法的架构。 AMP(Automatic Mixed Precision) 应运而生。2021年前还是使用 Apex

`from apex import amp`，现在 PyTorch 已经支持原生 `torch.amp`

![BF16 compared to FP32 and FP16](https://images.contentstack.io/v3/assets/blt71da4c740e00faaa/blt40c8ab571893763a/65f370cc0c744dfa367c0793/EXX-blog-fp64-fp32-fp-16-5_(3).jpg?format=webp)

## 混合精度训练机制

混合精度训练的核心矛盾是 **FP16/BF16 训练的更快，更省内存，但是数值范围太小，容易出现梯度下溢（变成0）导致模型无法正常训练**



Reference：https://arxiv.org/abs/1710.03740

![image-20260530131548100](https://github.com/Beater-0x7ff/PyTorch-SourceCode-Overview/blob/main/pytorch%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90/figures/mixed_prec_iteration.png)

AMP 的目标就是为用户提供了自动混合精度训练机制，**在保证模型精度基本不变的前提下，提高训练速度并降低显存占用**，AMP 主要有两个组件实现 `torch.amp.autocast` 和 `torch.amp.GradScaler`，前者负责自动选择计算精度，后者负责解决训练中梯度下溢问题



### autocast

`autocast` 可以作为上下文管理器使用

```python
with torch.amp.autocast(
    "cuda",
    dtype=torch.float16
):
    output = model(x)
    loss = criterion(output, target)
```

在 `autocast` 区域内，PyTorch 会根据算子的数值稳定性自动选择合适的数据类型。AMP 不会在每个迭代复制一份 FP16 模型。现在 PyTorch AMP 采用的是算子级别的自动精度调度。如上图来说，模型参数以 FP32 保存，计算过程是根据算子自动选择精度。



所谓 **自动选择精度**，实际上是 Python 只负责打开了开关，真正的精度选择发生在 C++ Dispatcher 的 Autocast dispatch key 中，具体流程如下：

1. **Python 入口 `with autocast()`**

   ```python
   with torch.amp.autocast("cuda", dtype=torch.float16):
       y = torch.matmul(x, w)
   ```

   进入 `with` 时，会调用

   ```python
   autocast.__enter__()
   ```

   它主要设置几个全局/线程局部状态

   ```python
   torch.set_autocast_enabled("cuda", True)
   torch.set_autocast_dtype("cuda", torch.float16)
   torch.autocast_increment_nesting()
   torch.set_autocast_cache_enabled(True)
   ```

   也就是说，Python 层只是告诉 PyTorch

   - 当前线程的 cuda autocast 已经开启
   - fast dtype  = float16

2. **调用算子：进入 Dispatcher**

   当执行 `y = torch.matmul(x, w)`

   Python 不会直接调用某个固定 CUDA kernel，而是进入 PyTorch 的 **Dispatcher**

   Dispatcher 会根据当前 Tensor 的信息和当前状态选择合适的 dispatch key

   例如 `AutocastCUDA`，`AutogradCUDA`

   因此此时 `autocast` 已经开启，所以会优先进入 `AutocastCUDA`

   官方 dispatcher 文档也会明确说，Autocast dispatch key 用来实现 AMP，autocast wrapper kernel 会在运行真正的 op(算子) 前，把输入 cast 到合适的精度

3. **AutocastCUDA wrapper：决定是否 cast**

   进入 `AutocastCUDA` 后，并不是马上执行真正的 `matmul`

   而是先进入一个 **autocast wrapper kernel**

   这个 wrapper 会根据算子策略决定 dtype （例如 matmul / linear / conv 适合低精度，于是 cast 输入到 float16 / bfload16）

   PyTorch 的 `autocast_mode.cpp` 正是这部分核心实现所在

4. **Redispatch：调用真正的 CUDA kernel**

   Autocast wrapper 完成 dtype 处理后，会再次把请求交给 Dispatcher

   但这次会排除 Autocast key，避免递归的调用自己

5. **退出 autocast**

   离开 `with` 块时调用

   ```python
   autocast.__exit__()
   ```

   它会恢复之前保存的状态

   ```python
   torch.set_autocast_enabled("cuda", prev)
   torch.set_autocast_dtype("cuda", prev_fastdtype)
   torch.set_autocast_cache_enabled(prev_cache_enabled)
   ```

   如果是退出最外层 autocast 还会清理缓存

   ```python
   torch.clear_autocast_cache()



### GradScaler

`GradScaler` 的作用就是在反向传播前给 loss 乘一个 scale factor，为了不影响学习率，在梯度更新前将梯度 unscale。用这种方案能防止梯度下溢。基本流程如下：

1. 模型参数通常仍以 FP32 保存。

2. 在每个 iteration：

   1. 使用 `autocast` 进行混合精度前向传播 
      部分算子使用 FP16/BF16，部分算子保留 FP32。

   2. 计算 loss。

   3. 使用 `GradScaler` 将 loss 乘以 scale factor $s$。

   4. 对 scaled loss 执行反向传播，得到 scaled gradient。

   5. 使用 unscaled gradient 更新模型参数

      

那么显然的，如何选择 scale factor $s$ 是一个非常重要的问题。设置为常数显然不是一个合适的选择，因为 scale factor 的设置需要防止 overflow 和 underflow。`GradScaler` 的策略非常简单：

每次 iteration 计算 scaled loss 然后反向传播后，检查有没有参数梯度 `inf` 或 `nan`

- 如果没有则每隔 `growth_interval` 次更新 `scale *= 2`
- 如果有先跳过 `optimizer.step()` 然后 `scale *= 0.5`





## AMP 模块 API 

示例

```python
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

x = torch.randn(64, 10).to(device)
y = torch.randint(0, 2, (64,)).to(device)

for step in range(5):
    optimizer.zero_grad()

    with torch.amp.autocast(
        device_type=device,
        dtype=torch.float16,
        enabled=(device == "cuda")
    ):
        pred = model(x)
        loss = loss_fn(pred, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"step={step}, loss={loss.item():.4f}")
```



### autocast 类

`autocast` 可以作为上下文管理器使用，用来指定某一段代码按自动混合精度执行

```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)
```

#### 

#### autocast 算子

在 `autocast` 区域内，PyTorch 会根据算子的数值稳定性自动选择 dtype。

通常适合低精度的算子包括：

- Linear
- Conv
- MatMul
- Batch MatMul

这些算子都可以使用 Tensor Core 加速

而对数值稳定性敏感的算子通常会保持 FP32，例如

- softmax
- layernorm
- 部分 reduction 操作



#### Mismatch Error

如果在 `autocast` 区域内得到的是低精度 Tensor，而区域外又和 FP32 Tensor 混合计算，可能需要手动转换：

```python
import torch

a = torch.randn(8, 8, device="cuda")
b = torch.randn(8, 8, device="cuda")
c = torch.randn(8, 8, device="cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    out_fp16 = torch.mm(a, b)

print(out_fp16.dtype)
out_fp32 = torch.mm(c, out_fp16.float())
print(out_fp32.dtype)

# torch.float16
# torch.float32
```





#### autocast 嵌套使用

可以在一个 `autocast` 区域中临时关闭混合精度

```python
import torch

a = torch.randn(8, 8, device="cuda")
b = torch.randn(8, 8, device="cuda")
c = torch.randn(8, 8, device="cuda")

with torch.amp.autocast("cuda", dtype=torch.float16):
    x = torch.mm(a, b)
    print(x.dtype)

    with torch.amp.autocast("cuda", enabled=False):
        y = torch.mm(c, x.float())
        print(y.dtype)

    z = torch.mm(x, y)
    print(z.dtype)
    
# torch.float16
# torch.float32
# torch.float16
```



#### autocast 作为装饰器

`autocast` 也可以作为装饰器使用

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    @torch.amp.autocast("cuda", dtype=torch.float16)
    def forward(self, x):
        return self.linear(x)
```



#### autocast 自定义函数

对于自定义 `torch.autograd.Function` 可以使用 

```python
torch.amp.custom_fwd
torch.amp.custom_bwd
```

示例

```python
class MyMM(torch.autograd.Function):

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return a.mm(b)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad.mm(b.t()), a.t().mm(grad)
    
    
mymm = MyMM.apply
with torch.amp.autocast("cuda", dtype=torch.float16):
    output = mymm(input1, input2)
```





#### autocast 源码分析

```python
def __enter__(self):
    # TorchScript 脚本模式下的特殊处理
    if torch._jit_internal.is_scripting():
        if self.fast_dtype is None:
            raise AssertionError("fast_dtype must not be None in scripting mode")
        return self

    # 保存进入 autocast 前的状态，方便 __exit__ 恢复
    self.prev_cache_enabled = torch.is_autocast_cache_enabled()
    self.prev = torch.is_autocast_enabled(self.device)
    self.prev_fastdtype = torch.get_autocast_dtype(self.device)

    # 开启/关闭指定 device 上的 autocast
    torch.set_autocast_enabled(self.device, self._enabled)

    # 设置 autocast 使用的低精度类型
    # 例如 cuda 上通常是 float16 或 bfloat16
    torch.set_autocast_dtype(self.device, self.fast_dtype)

    # 记录 autocast 嵌套层数
    torch.autocast_increment_nesting()

    # 设置 autocast cache 是否启用
    torch.set_autocast_cache_enabled(self._cache_enabled)

    # 如果处于 TorchFunctionMode，例如 FX / ProxyTensor
    # 需要通知对应 mode 进入 autocast 状态
    if torch._C._is_torch_function_mode_enabled():
        stacks = torch.overrides._get_current_function_mode_stack()

        for mode in stacks:
            if isinstance(
                mode,
                torch.fx.experimental.proxy_tensor.PreDispatchTorchFunctionMode,
            ):
                args = (
                    self.device,
                    self.fast_dtype,
                    self._enabled,
                    self._cache_enabled,
                )

                # 通知 ProxyTensor/FX tracing 系统进入 autocast
                mode.__torch_function__(torch.amp._enter_autocast, (), args)

                return self

    # 返回上下文管理器对象
    return self
```



```python
def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):

    # TorchScript 脚本模式下不做额外处理
    if torch._jit_internal.is_scripting():
        return

    # autocast 嵌套层数 -1
    # 如果已经退出最外层 autocast，就清空 autocast cache
    if torch.autocast_decrement_nesting() == 0:
        torch.clear_autocast_cache()

    # 恢复进入 autocast 前的 enabled 状态
    torch.set_autocast_enabled(self.device, self.prev)

    # 恢复进入 autocast 前的 dtype
    torch.set_autocast_dtype(self.device, self.prev_fastdtype)

    # 恢复进入 autocast 前的 cache 设置
    torch.set_autocast_cache_enabled(self.prev_cache_enabled)

    # 如果处于 TorchFunctionMode，例如 FX / ProxyTensor
    # 通知 tracing 系统退出 autocast
    if torch._C._is_torch_function_mode_enabled():
        stacks = torch.overrides._get_current_function_mode_stack()

        for mode in stacks:
            if isinstance(
                mode,
                torch.fx.experimental.proxy_tensor.PreDispatchTorchFunctionMode,
            ):
                mode.__torch_function__(torch.amp._exit_autocast, (), ())
                return False

    # 返回 False 表示不吞掉异常
    return False
```



```python
def __call__(self, func):
    # TorchScript 脚本模式下直接返回原函数
    if torch._jit_internal.is_scripting():
        return func

    # autocast 作为装饰器使用时，传入对象必须是可调用对象
    if not callable(func):
        raise TypeError(
            f"autocast()(func) requires a callable, but got {type(func).__name__}. "
            f"Did you mean to use autocast as a context manager? For example:\n"
            f"    with torch.autocast(device_type=...):\n"
            f"        output = model(input)"
        )

    # 将 func 包装成带 autocast 上下文的函数
    return autocast_decorator(self, func)
```







### GradScaler 类

`GradScaler` 用于 FP16 训练中的动态 loss scaling

```python
torch.amp.GradScaler(
    "cuda",
    
    # 初始 scale factor
    init_scale=65536.0,
    
    # 没有溢出时的增长系数
    growth_factor=2.0,
    
    # 出现溢出时的下降系数
    backoff_factor=0.5,

    # 连续多少次无溢出后增长 scale
    growth_interval=2000,
    
    # 是否启用 GradScaler
    enabled=True
)
```



#### scale(output) 方法

`scale()` 会把 loss 乘以当前 scale factor

```python
scaled_loss = scaler.scale(loss)
scaled_loss.backward()
```



#### unscale_(optimizer) 方法

`unscale_()` 会把梯度除以 scale factor，恢复真实梯度

通常用于梯度裁剪前

```python
scaler.scale(loss).backward()

scaler.unscale_(optimizer)

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scaler.step(optimizer)
scaler.update()
```

如果不需要手动操作梯度，可以不显式调用 `unscale_()`，因为 `scaler.step(optimizer)` 内部会自动处理



#### step(optimizer) 方法

该方法主要完成了两件事：

1. 如果还没有手动 `unscale_()`，则自动 unscale 梯度
2. 检查梯度中是否有 `inf` 或 `nan`:
   - 如果没有则执行 `optimizer.step()`
   - 如果有，则跳过本次参数更新

```python
scaler.step(optimizer)
```



#### update(new_scale=None) 方法

`update()` 用于更新 scale factor

```python
scaler.update()
```

如果连续多次没有溢出

```python
scale *= growth_factor
```

如果发现 `inf` 或 `nan`

```python
scale *= backoff_factor
```

这就是 **动态 loss scaling**