# 功能验证记录

对 triton-ng 在真实 Triton 服务器上的功能验证。记录日期 2026-11-02，被测提交 `6a009b4`（Memory manager #6），
`deps/` 指向 core/common 的 `r25.09` 分支。

## 环境前置

脚本 `scripts/verify.sh` 优先使用隔离安装的 rustup（若在 `~/.local/share/cargo/bin` 下），找不到 libclang 时直接报错。
下面两处依赖是当时补齐的，供复现参考。

| 依赖 | 现象 | 处理 |
|---|---|---|
| libclang | Homebrew 的 llvm 23 需要 GLIBC 2.38，Ubuntu 22.04 的 glibc 2.35 加载失败，bindgen 在构建期 panic | 使用 PyPI `libclang` 18.1.1 的 manylinux wheel：`pip download libclang --no-deps -d <目录>` 后解开，`LIBCLANG_PATH` 指向 `clang/native` |
| Rust 工具链 | Homebrew rust 1.99.1 的 `libstd.so` 需要 GLIBC 2.39，proc-macro crate 的测试二进制无法启动，`make tests` 在 `triton-ng-macros` 处中断 | 用 rustup 安装 1.99.0 到隔离目录（`RUSTUP_HOME`、`CARGO_HOME` 指向 `~/.local/share`），`--no-modify-path` 不改动 PATH |

`libtriton_custom_backend.so` 只依赖 `libc`、`libm`、`libgcc_s`，libstd 静态链接，可以直接挂进 Triton 容器。

## 导出符号

`nm -D --defined-only` 输出中只有 7 个 `TRITONBACKEND_*`，与 `tritonbackend.h` 要求的必需集合一致：

```
TRITONBACKEND_Initialize  TRITONBACKEND_Finalize
TRITONBACKEND_ModelInitialize  TRITONBACKEND_ModelFinalize
TRITONBACKEND_ModelInstanceInitialize  TRITONBACKEND_ModelInstanceFinalize
TRITONBACKEND_ModelInstanceExecute
```

没有 `TRITONBACKEND_ApiVersion`，Triton 因此按 API 1.0 处理（服务器日志里 onnxruntime 与 identity 都打印
`TRITONBACKEND API version: 1.19`，Rust 后端没有版本行）。没有 `TRITONBACKEND_GetBackendAttribute`，
无法声明 preferred instance groups、并行加载实例等属性。

## 测试结果

服务器用 `scripts/verify.sh` 起：挂载 `target/release/libtriton_custom_backend.so` 与 `verify/models`
（`mnist` 为 `instance_group count: 2`、`worker_threads: "3"`、`max_batches_in_flight: "4"`、动态批处理）。

| 项目 | `25.09-py3` | `26.09-py3`（服务器版本 2.73.0） |
|---|---|---|
| 后端加载 | 成功 | 成功 |
| 模型状态 | `mnist`、`mnist_onnx`、`identity_gpu` 全部 READY | 同 |
| `make tests` | 64/64 通过 | 64/64 通过（加 `--allow-client-shm=true`） |
| `make tests`（服务器默认参数） | — | 61/64，见下 |
| 示例客户端 | `DIGIT: shape=[1, 1] values=Int32([5])` | 同 |
| `make lint` | 无告警 | — |

64 个用例由 34 个 `triton-ng` 单元测试、2 个客户端单元测试、28 个需要真实服务器的集成测试组成。集成测试覆盖：

- BLS 链路：后端收到请求后异步调用 `mnist_onnx`，取 argmax 后返回 `DIGIT`
- 动态批处理下的并发请求
- 取消：`Request::is_cancelled` 与 `respond_cancelled`
- 内存缓冲：RAM、pinned、GPU 三类缓冲的分配、复用、作为输出缓冲；一个 `Arc<GpuBuffer>` 同时作为两个 BLS 请求的输入
- 错误路径：输出写入尺寸不符的缓冲
- 模型卸载：持有中的请求答复完成后才卸载
- 客户端：元数据、配置、统计、仓库索引、加载卸载、系统与 GPU 共享内存、trace、log

### 26.09 的默认值变化

不加 `--allow-client-shm=true` 时，3 个共享内存用例失败，服务器返回：

```
Client shared memory is disabled. Start the server with '--allow-client-shm=true' to enable.
```

失败用例：`system_shared_memory_status_is_empty_when_no_regions_are_registered`、
`system_shared_memory_region_can_be_registered_and_unregistered`、
`cuda_shared_memory_status_is_empty_when_no_regions_are_registered`。这是 26.x 关闭了客户端共享内存的默认值，
补上该参数后 64/64 全部通过。

### 独立交叉核对

示例客户端向 `mnist`（Rust 后端）发全零输入得到 `DIGIT=5`。直接向 `mnist_onnx` 发同样的输入，logits 为
`[-0.0449, 0.0078, 0.0681, 0.03, -0.1264, 0.1402, -0.0553, -0.0494, 0.0843, -0.0545]`，argmax 也是 5。
转发与取最大值的链路结果一致。`scripts/cross_check.py` 固化这条核对。

## 能力边界

读导出符号与源码确认，以下 Triton 后端接口未被使用：

| 接口 | 影响 |
|---|---|
| `TRITONBACKEND_ResponseFactory` | 不支持 decoupled 响应，无法实现流式后端（逐 token 返回） |
| `TRITONBACKEND_State` | 不支持 sequence batching 与隐式状态 |
| `TRITONBACKEND_ApiVersion`、`TRITONBACKEND_GetBackendAttribute` | Triton 按 API 1.0 处理，无法声明后端属性 |

`#[triton_ng::backend]` 生成的 `extern "C"` 入口在进入用户代码前套了 `catch_unwind`，用户代码 panic 会转成
`TRITONSERVER_Error`，不会跨 FFI 边界展开终止服务器进程。

## 遗留风险

- 验证机是 WSL2，宿主驱动 610.60；`26.09-py3` 容器启动时报
  `built for NVIDIA Driver Release 615.71 or later, but version 610.60 was detected and compatibility mode is UNAVAILABLE`。
  本次 GPU 相关用例全部通过、日志无 CUDA 错误，但驱动版本不满足该镜像要求，换机器需要重新确认 GPU 路径。
- 未测吞吐与延迟，本记录只覆盖功能。
- 未测多 GPU、跨设备实例、多模型共享同一后端对象的并发加载。

## 复现

```
scripts/verify.sh 26.09-py3     # 默认
scripts/verify.sh 25.09-py3
```

两个标签都跑过，退出码 0。脚本依次完成：检测工具链与 libclang、`make build`、校验导出符号、下载 MNIST 模型、组装
`target/verify-repo`、必要时拉镜像、读镜像自己的 `--help` 决定是否加 `--allow-client-shm=true`、起容器等待就绪、
`make tests`、运行示例客户端、执行 `scripts/cross_check.py`。失败时保留容器并打印日志尾部。
