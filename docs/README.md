# triton-ng documentation

How to write custom Triton backends with triton-ng. Start with the [main README](../README.md)
for a minimal backend, the gRPC client and running the example.

| Section | What it covers |
|---|---|
| [Execution model and thread safety](execution_model_and_thread_safety.md) | Instances and their runtimes, state of models and instances, request ownership, response streams, BLS futures and streams, threads, cancellation, shutdown, batching |
| [Model parameters](model_parameters.md) | `worker_threads` and `max_batches_in_flight` in the model config |
| [Memory buffers](memory_buffers.md) | RAM, pinned and GPU buffers from Triton's memory manager for BLS inputs and outputs, caching them in a backend |

[`example/custom-backend`](../example/custom-backend/src/lib.rs) contains the MNIST backend and
the `decoupled` / `stream_proxy` models used to demonstrate response and BLS streams.
