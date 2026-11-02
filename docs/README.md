# triton-ng documentation

How to write custom Triton backends with triton-ng. Start with the [main README](../README.md)
for a minimal backend, the gRPC client and running the example.

| Section | What it covers |
|---|---|
| [Execution model and thread safety](execution_model_and_thread_safety.md) | Instances and their runtimes, state of models and instances, request ownership, BLS futures, threads, cancellation, shutdown, batching |
| [Model parameters](model_parameters.md) | `worker_threads` and `max_batches_in_flight` in the model config |
| [Memory buffers](memory_buffers.md) | RAM, pinned and GPU buffers from Triton's memory manager for BLS inputs and outputs, caching them in a backend |
| [Verification](verification.md) | Test run against a live Triton server: results on `25.09-py3` and `26.09-py3`, capability boundaries, known risks, how to reproduce with `scripts/verify.sh` (written in Chinese) |

[`example/custom-backend`](../example/custom-backend/src/lib.rs) is the MNIST example backend that
uses all of it.
