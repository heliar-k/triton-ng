# Rust 后端 SDK 缺失能力的实现计划

状态：待实施
依据：`deps/` 指向 Triton `r25.09`（服务器版本 2.61.0），`tritonbackend.h` 的 API 版本 1.19，`tritonserver.h` 的 API 版本 1.34。文中出现的 C 函数名都以这两个头文件为准。

## 现状

`triton-ng` 目前封装并调用了头文件里 130 个 `TRITONBACKEND_*` 符号中的 42 个函数。已有的能力：七个后端入口、请求与响应的基本读写、响应参数、BLS（`TRITONSERVER_ServerInferAsync` 加自己实现的 `TRITONSERVER_ResponseAllocator`）、内存管理器提供的 RAM / pinned / GPU 缓冲、每个实例一个 tokio runtime、`worker_threads` 与 `max_batches_in_flight` 两个模型参数。

尚未封装的部分集中在：流式响应、序列状态、自定义批处理、host policy、指标与 trace、后端属性、输入输出的数据面、请求元数据、BLS 请求构造。

## Problem Statement

用 triton-ng 写后端时，有一批接口能表达的能力表达不出来：

- 一个请求只能得到一条响应，逐块输出的模型无法代理；
- 序列模型没有状态可读写，`sequence_batching` 的配置没有意义；
- 指标端点里这个后端的 `nv_inference_*` 全是空的，trace 里看不到后端事件；
- 请求只能读 CPU 内存并且必然拷贝一次，输出只能写 CPU 内存；
- 下游模型读的请求参数设不进去，所以代理 tensorrtllm 一类模型时无法传 `max_tokens`；
- 模型配置省略 `instance_group` 时无法声明自己需要的实例数量与设备。

要绕过这些限制，后端作者只能在 `unsafe` 里直接调 FFI，或者放弃这些能力。

## Solution

按七个阶段把这些接口补齐，每个阶段都能单独发布。第一个阶段重做请求与响应的生命周期，因为流式响应、统计上报、序列状态、trace 四件事都取决于「request 什么时候被释放」。其余六个阶段都是新增，不改动已有类型的语义。

## User Stories

1. 作为后端作者，我想让一个请求返回多条响应并在最后一条上标记结束，以便代理一个逐块输出的模型。
2. 作为后端作者，我想在响应发完之前释放请求，以便 Triton 早一点回收输入张量占用的内存。
3. 作为后端作者，我想在请求释放之后仍然能知道客户端是否取消，以便停止正在进行的流。
4. 作为后端作者，我想从下游 decoupled 模型逐条收到结果，以便边收边转发。
5. 作为后端作者，我想给每条响应上报开始与结束时间，以便流式响应的耗时出现在指标里。
6. 作为后端作者，我想上报每个请求的成功与耗时，以便 `nv_inference_count` 与 `nv_inference_*_duration_us` 不再为空。
7. 作为后端作者，我想上报模型与实例占用的内存，以便运维看到显存占用。
8. 作为后端作者，我想在请求上记录 trace 事件，以便定位慢在哪一段。
9. 作为后端作者，我想读请求的相关性 ID 与序列标志，以便实现有状态的模型。
10. 作为后端作者，我想读写序列状态，以便实现 `sequence_batching` 模型。
11. 作为后端作者，我想给下游 BLS 请求设置参数，以便把 `max_tokens` 这类信息传给下游模型。
12. 作为后端作者，我想给下游 BLS 请求设置相关性 ID、序列标志与优先级，以便把请求转给有状态的模型并按优先级排队。
13. 作为后端作者，我想从 BLS 请求里移除输入或输出，以便复用同一个请求对象。
14. 作为后端作者，我想在 BLS 请求里指定 host policy，以便和多 host 内存配置配合。
15. 作为后端作者，我想直接读输入缓冲区而不复制，以便处理大张量时少一次拷贝。
16. 作为后端作者，我想读 GPU 上的输入，以便把数据直接交给下一段 GPU 处理。
17. 作为后端作者，我想把输出直接写进 GPU 内存，以便结果留在显存里。
18. 作为后端作者，我想用 fp16 / bf16 / int8 的读写方法，以便不用自己写解码。
19. 作为后端作者，我想声明首选的 instance group，以便模型配置省略 `instance_group` 时得到需要的实例数量与设备。
20. 作为后端作者，我想声明实例可以并行加载，以便多实例模型加载得更快。
21. 作为后端作者，我想在 `initialize` 里读到后端的配置与自身所在目录，以便定位随后端一起分发的文件。
22. 作为后端作者，我想在 `initialize` 里拿到内存管理器，以便在模型加载之前分配缓冲。
23. 作为后端作者，我想在后端上存一份共享状态，以便所有模型共用。
24. 作为后端作者，我想自己决定哪些请求放进同一批，以便实现 in-flight batching。
25. 作为后端作者，我想知道服务器支持的后端 API 版本，以便版本不兼容时在加载阶段失败。
26. 作为后端作者，我想读实例的 host policy 与 profile 设置，以便按部署配置调整参数。
27. 作为后端作者，我想把最终响应推迟到所有批处理工作结束时，以便在最后一个响应之前更新序列状态。
28. 作为客户端作者，我想用一条流式连接发送请求，以便接收流式响应。
29. 作为维护者，我想每个新接口都有文档与示例，以便使用者不需要读 C 头文件。

## Implementation Decisions

### 阶段 0：API 版本校验

在 `Backend::initialize` 之前调用 `TRITONBACKEND_ApiVersion`，与 bindgen 已经从 `tritonbackend.h` 生成的两个常量 `TRITONBACKEND_API_VERSION_MAJOR` / `MINOR` 比较。主版本不同，或者服务器的小版本小于编译期的小版本，就让 `initialize` 返回错误，由 Triton 拒绝加载。

没有降级路径：版本不匹配直接失败，失败信息里带上双方的版本号。

### 阶段 1：请求与响应的生命周期，双向流式

涉及模块：`request`、`response`、`input`、`output`、`server`、`inference_request`、`inference_future`、`inference_result`、`response_allocator`、`executor`。

**决定 1：把 `Request` 分成拥有所有权的 `Request` 和借用的 `RequestView<'a>`。**
所有读取操作实现在 `RequestView` 上：`input_names`、`output_names`、`get_input`、`parameter`、`is_cancelled`、`id`、`flags`、`correlation_id`、`timeout`。`Request` 内含一个视图，另外持有所有权与释放逻辑，`Request::view()` 给出视图。

这样分的依据是 `ModelBatchIncludeRequest`：它收到的 request 由 Triton 拥有，后端只能读，绝对不能释放。

**决定 2：响应的所有权与请求分离。**
`Response` 不再持有 `Request`，改为借用响应的创建者。`Output<'_>` 继续借用 `Response`，这条借用链在编译期保证头文件要求的「同一个请求的响应必须串行创建输出」。

`Response::new(request)` 保留，接管请求所有权并发送最终响应，现在的用法不变。新增 `ResponseFactory` 走流式：

```rust
let mut factory = ResponseFactory::new(request)?;   // 接管请求的所有权

let mut response = factory.response()?;             // 借 &mut factory，同一时刻只有一条响应在写
response.create_output("TOKENS", DataType::Int64, &[1, n])?.write_i64_vec(&tokens)?;
response.send()?;                                    // 不带 FINAL 标志

factory.finish()?;                                   // 发 FINAL 并释放请求
```

- `factory.finish()` 走 `TRITONBACKEND_ResponseFactorySendFlags` 发 FINAL，然后释放请求，再删工厂。发了 FINAL 之后不能再创建响应。
- `factory.release_request()` 允许提前释放请求，之后工厂仍可用于 `is_cancelled()`（走 `TRITONBACKEND_ResponseFactoryIsCancelled`）与继续发送剩余响应。
- 「是否已经有最终响应」由请求与工厂共享，因为响应对象不再持有请求。`Request` 的 Drop 兜底逻辑保留：没有最终响应就补一条错误响应，然后释放。
- `ResponseFactory` 的 Drop：未结束时补一条错误响应（请求还在的情况下），发 FINAL，释放请求，删工厂。这条兜底路径必须走完，因为删掉工厂之后 Triton 对未结束的请求没有定义行为。

**决定 3：BLS 方向分成两个入口，共用一条回调路径。**

- `Server::infer_async(request)` 保持现有语义（聚合成一个最终结果），内部改成消费流。
- 新增 `Server::infer_stream(request)`，返回 `InferenceStream`，`async fn next()` 逐条给出 `InferenceResult`。
- 响应回调把每条响应转成一个结果送进 channel，收到 FINAL 时关闭 channel。
- `InferenceResult` 增加 `is_final` 字段。
- `infer_stream` 在提交前拒绝带预分配输出缓冲的请求：同一块缓冲会被多条响应重复写入，提交时失败比收到第二块时报错更早暴露问题。

**决定 4：`Context` 里放一个可跨 `.await` 的实例句柄。**
统计上报与 trace 都需要 `TRITONBACKEND_ModelInstance*`，而 `ModelInstance` 既不是 `Send` 也不是 `Sync`、只在钩子里有效。新的句柄类型是 `Send + Sync + Clone`，有效性论证与 `Server` 相同：`model_instance_finalize` 之前 triton-ng 会等所有批处理结束，因此批处理任务持有期间实例指针有效。

**决定 5：请求记录时间戳。**
收到批处理时记 `exec_start`，批处理的 future 真正开始执行时记 `compute_start`，响应发出时记结束时间。这几项是阶段 2 的输入，用同一个时钟源的纳秒时间戳，跨请求只做差值。

### 阶段 2：指标、trace、内存用量

- 请求级：`TRITONBACKEND_ModelInstanceReportStatistics`，挂在「请求完成且尚未释放」这一点上，由请求的结束路径触发，上报成功与否和四个时间戳。
- 响应级：`ModelInstanceResponseStatisticsNew` → `SetResponseFactory`（一次性响应走 `SetModelInstance`）→ `SetResponseStart` → `SetComputeOutputStart` → `SetResponseEnd` → `SetError` → `ModelInstanceReportResponseStatistics` → `Delete`。每条响应一组，与阶段 1 的发送路径放在同一处，流式下每块各一组，只有最后一块带结束时间。
- 内存用量：先封装 `TRITONSERVER_BufferAttributesNew` / `SetMemoryType` / `SetMemoryTypeId` / `SetByteSize` / `Delete`，再暴露 `ModelInstanceReportMemoryUsage` 与 `ModelReportMemoryUsage`，调用点由后端自己在 `model_instance_finalize` 与 `model_finalize` 之前决定。
- trace：`TRITONBACKEND_RequestTrace` 给出 trace 的借用视图，`report_activity(activity, timestamp_ns)` 写事件。Triton 没有开启 trace 时该调用返回空指针，Rust 侧给出 `None`。
- 请求级与响应级指标默认上报，与官方后端一致；trace 只读不建。

### 阶段 3：加载属性、请求元数据、BLS 请求构造、类型读写

- `BackendHandle` 增加 `config()`（`TRITONBACKEND_BackendConfig`）、`artifacts()`（`TRITONBACKEND_BackendArtifacts`）、`memory_manager()`（`TRITONBACKEND_BackendMemoryManager`）、`set_state` / `state`（`BackendSetState` / `BackendState`）、`attributes()`（`GetBackendAttribute` 之后设置首选的 instance group 与并行加载）。
  `Backend` trait 不新增方法：`initialize` 已经拿得到 `BackendHandle`，属性通过它设置。
- `RequestView` 增加 `id`、`correlation_id`、`correlation_id_string`、`flags`、`timeout`、`trace` 的读取。`flags` 给出 `SEQUENCE_START` 与 `SEQUENCE_END` 两个布尔值，不直接暴露 C 位掩码。
- `InferenceRequest` 增加 `set_parameter`（布尔、整数、浮点、字符串四种）、`set_correlation_id`、`set_correlation_id_string`、`set_flags`、`set_timeout`、`set_priority`、`remove_input`、`remove_requested_output`、`append_input_data_with_host_policy`。
  `remove_input` 与 `remove_requested_output` 必须同步清理请求里的 `buffers` 与 `outputs` 记账，否则释放后仍持有已被 Triton 遗忘的缓冲。
- `Server` 增加后端侧会用到的读接口：`model_config`（`TRITONSERVER_ServerModelConfig`）、`model_statistics`（`TRITONSERVER_ServerModelStatistics`）、`model_index`（`TRITONSERVER_ServerModelIndex`）。
- `Input` 与 `Output` 补 fp16、bf16、fp64、int8、uint8、uint16、uint32、bool 的读写方法。fp16 与 bf16 用 `u16` 位模式承载，不引入半精度依赖，方法文档里写明位的含义。

### 阶段 4：输入输出的数据面

- 输入：`Input::buffers()` 返回逐块的借用视图，每块带 `memory_type` 与 `memory_type_id`。主机内存给出 `&[u8]`，设备内存给出指针、长度与设备号，读取设备内存的方法是 `unsafe` 并在文档里写明调用方要承担的约束。`as_bytes` 保留作为拼接拷贝的便捷路径，内部的拼接函数变成私有。
- host policy 的输入侧与 `ModelInstanceHostPolicy` 一起做：从实例读 host policy 名称，再按名称取输入属性与缓冲（`InputPropertiesForHostPolicy` / `InputBufferForHostPolicy`）。
- 输出：`Response::create_output` 增加一个指定内存位置的变体，向 `TRITONBACKEND_OutputBuffer` 请求期望的 `memory_type` 与 id；Triton 返回的值可能与请求不同，不一致时按现有风格失败（现有代码在返回 GPU 内存时已经这么做）。再加一个写设备缓冲区的 `unsafe` 方法。

### 阶段 5：序列与有状态模型

- 新增 `State` 类型：`State::new(view, name, dtype, shape)` → `write_bytes` → `update`。`update` 消费 `State`，对应 `TRITONBACKEND_StateUpdate`。
- `State::new` 走 `TRITONBACKEND_StateNew`，要求请求尚未释放；因此这一步要用阶段 1 提供的「响应发完之后仍能持有请求」路径。
- `flags` 的语义在阶段 3 已经给出，这里只需要在示例里演一遍首请求与末请求的处理。

### 阶段 6：自定义批处理

核对结论：Triton 25.09 的自定义批处理是一个**独立的共享库**，不是后端库里的符号。`TritonModel::SetBatchingStrategy` 从模型参数 `TRITON_BATCH_STRATEGY_PATH` 取路径，取不到就在后端搜索路径下找 `batchstrategy.so`；五个符号全部存在才启用，缺一个就报错；与 `sequence_batching` 互斥。

因此这一阶段产出一个独立的 crate，编译为 cdylib，宏 `#[batcher]` 导出五个符号：

- `TRITONBACKEND_ModelBatcherInitialize` / `ModelBatcherFinalize`：模型级只读状态，模型加载时建、卸载时删。
- `TRITONBACKEND_ModelBatchInitialize` / `ModelBatchFinalize`：每批一组临时状态。
- `TRITONBACKEND_ModelBatchIncludeRequest`：接到的是 `RequestView`，返回是否收进这一批。

trait 的形状：一个 `Batching` trait，关联类型 `Batch: Default` 表示一批的临时状态，三个方法分别对应批开始、决定收不收、批结束。宏与 `#[backend]` 共用输出替换与错误转换的实现。

### 阶段 7：零碎属性与客户端流式

- `ModelInstance` 增加 `profile_count` / `profile_name`、`is_passive`、`secondary_device_properties`。
- `Model` 增加 `auto_complete_config`（`TRITONBACKEND_ModelAutoCompleteConfig`）与 `set_config`（`TRITONBACKEND_ModelSetConfig`）。
- 客户端增加 `TritonClient::stream_infer`：一条双向流，请求按顺序发出，响应按顺序读回。

## 阶段与阻塞关系

| 阶段 | 内容 | 阻塞于 |
|---|---|---|
| 0 | API 版本校验 | 无 |
| 1 | 请求与响应生命周期、双向流式 | 0 |
| 2 | 指标、trace、内存用量 | 1 |
| 3 | 加载属性、请求元数据、BLS 请求构造、类型读写 | 0 |
| 4 | 输入输出数据面 | 1（借用视图）、3（host policy 名称） |
| 5 | 序列与有状态模型 | 1（延后释放） |
| 6 | 自定义批处理 | 1（只读视图） |
| 7 | 零碎属性与客户端流式 | 0 |

阶段 2 与阶段 3 可以并行。

## Testing Decisions

- 单元测试放在被测代码所在文件末尾的 `#[cfg(test)] mod tests`。库的单元测试不链接 libtritonserver，凡是会触及 Triton 符号的逻辑都要先抽到纯 Rust 类型上（现有的 `element_count`、`InstanceExecutor`、`InferenceResult` 就是这样测的）。
- 需要真实服务器的集成测试放在各 crate 的 `tests/` 目录，用 `make tests` 跑，前置条件是 `make docker-env-up`。
- 新增两条测试接缝，都是新的示例模型：
  - 一个 decoupled 模型，逐块输出，同时验证阶段 1 的两端与阶段 2 的响应级指标；
  - 一个 `sequence_batching` 模型，验证阶段 5。
- 自定义批处理需要一个 `batchstrategy.so` 加一份引用它的模型配置，还需要一条新的 `make` 目标来构建这个库。新增目标要先约定。
- 端到端沿用 `scripts/verify.sh`（参数是镜像标签）与 `scripts/cross_check.py`。
- 每个阶段的验收：`make lint` 与 `make tests` 无告警无失败，加上该阶段自己的端到端现象（流式看到多块、指标端点看到计数、trace 输出里看到事件）。

## Out of Scope

- host policy 在语义层面的完整支持，只在读取层面接入。
- passive instance 与 secondary device 的完整支持，只暴露读接口。
- 客户端除 `ModelStreamInfer` 之外的 gRPC 扩展。
- Windows 上的导出符号处理。
- Triton 26.x 新增的接口。

## Further Notes

- 阶段 1 是唯一会改动现有公开 API 语义的一步，之后六个阶段都是新增；`Backend::model_instance_execute` 的签名在这之后保持稳定。
- 阶段 1 完成后要同步的文档：`docs/execution_model_and_thread_safety.md` 的句柄表与所有权说明、`README.md` 的示例、`docs/README.md` 的索引。
- 阶段 4 完成后要同步 `docs/memory_buffers.md` 里的缓冲流程图。
- 判定「哪些请求属于同一批」由 Triton 负责，阶段 6 只是给 Triton 一个附加条件，不改变 `dynamic_batching` 的排队规则。
