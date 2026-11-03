use std::str::FromStr;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use triton_ng::backend::Backend;
use triton_ng::{
    BackendHandle, BufferAttributes, Context, DataType, Error, GpuBuffer, InferenceFuture,
    InferenceRequest, MemoryManager, MemoryType, Model, ModelInstance, Parameter, PinnedRamBuffer,
    RamBuffer, Request, Response, ResponseFactory, UninitGpuBuffer,
};

const IMAGE: &str = "Input3";
const IMAGE_SHAPE: [i64; 4] = [1, 1, 28, 28];
const LOGITS: &str = "Plus214_Output_0";
const LOGITS_BYTES: usize = 10 * 4;
const IDENTITY_INPUT: &str = "INPUT0";
const IDENTITY_OUTPUT: &str = "OUTPUT0";

#[derive(Clone, Copy)]
enum OutputMemory {
    Ram,
    Pinned,
    Gpu,
    GpuInput,
    WrongSize,
}

impl FromStr for OutputMemory {
    type Err = Error;

    fn from_str(memory: &str) -> Result<Self, Error> {
        match memory {
            "ram" => Ok(Self::Ram),
            "pinned" => Ok(Self::Pinned),
            "gpu" => Ok(Self::Gpu),
            "gpu_input" => Ok(Self::GpuInput),
            "wrong_size" => Ok(Self::WrongSize),
            other => Err(format!("unknown memory '{other}'").into()),
        }
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Mnist,
    Decoupled,
    StreamProxy,
}

impl FromStr for Mode {
    type Err = Error;

    fn from_str(mode: &str) -> Result<Self, Error> {
        match mode {
            "mnist" => Ok(Self::Mnist),
            "decoupled" => Ok(Self::Decoupled),
            "stream_proxy" => Ok(Self::StreamProxy),
            other => Err(format!("unknown backend mode '{other}'").into()),
        }
    }
}

struct State {
    mode: Mode,
    target_model: String,
    identity_model: String,
    held: AtomicI64,
    cancelled: AtomicI64,
}

static CANCELLED_STREAMS: AtomicI64 = AtomicI64::new(0);

#[derive(Default)]
struct Buffers {
    ram: Mutex<Vec<RamBuffer>>,
    pinned: Mutex<Vec<PinnedRamBuffer>>,
    gpu: Mutex<Vec<GpuBuffer>>,
    ram_allocated: AtomicI64,
    pinned_allocated: AtomicI64,
    gpu_allocated: AtomicI64,
}

impl Buffers {
    fn ram(&self, memory: &MemoryManager, len: usize) -> Result<RamBuffer, Error> {
        let mut cached = self.ram.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached.iter().position(|buffer| buffer.len() == len) {
            Some(index) => Ok(cached.swap_remove(index)),
            None => {
                self.ram_allocated.fetch_add(1, Ordering::SeqCst);
                memory.ram(len)
            }
        }
    }

    fn pinned(&self, memory: &MemoryManager, len: usize) -> Result<PinnedRamBuffer, Error> {
        let mut cached = self.pinned.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached.iter().position(|buffer| buffer.len() == len) {
            Some(index) => Ok(cached.swap_remove(index)),
            None => {
                self.pinned_allocated.fetch_add(1, Ordering::SeqCst);
                memory.pinned(len)
            }
        }
    }

    fn gpu(
        &self,
        memory: &MemoryManager,
        device_id: i32,
        len: usize,
    ) -> Result<UninitGpuBuffer, Error> {
        let mut cached = self.gpu.lock().map_err(|_| "buffer cache is poisoned")?;
        match cached
            .iter()
            .position(|buffer| buffer.len() == len && buffer.device_id() == device_id)
        {
            Some(index) => Ok(cached.swap_remove(index).into_uninit()),
            None => {
                self.gpu_allocated.fetch_add(1, Ordering::SeqCst);
                memory.gpu(device_id, len)
            }
        }
    }

    fn put_ram(&self, buffer: RamBuffer) -> Result<(), Error> {
        self.ram
            .lock()
            .map_err(|_| "buffer cache is poisoned")?
            .push(buffer);
        Ok(())
    }

    fn put_pinned(&self, buffer: PinnedRamBuffer) -> Result<(), Error> {
        self.pinned
            .lock()
            .map_err(|_| "buffer cache is poisoned")?
            .push(buffer);
        Ok(())
    }

    fn put_gpu(&self, buffer: GpuBuffer) -> Result<(), Error> {
        self.gpu
            .lock()
            .map_err(|_| "buffer cache is poisoned")?
            .push(buffer);
        Ok(())
    }
}

struct MnistBackend;

#[triton_ng::backend]
impl Backend for MnistBackend {
    fn initialize(backend: &BackendHandle) -> Result<(), Error> {
        println!("Initializing backend: {}", backend.name()?);
        Ok(())
    }

    fn model_initialize(model: &Model) -> Result<(), Error> {
        model.set_state(State {
            mode: model.config_parameter_or("mode", Mode::Mnist)?,
            target_model: model.config_parameter_or("target_model", "mnist_onnx".to_string())?,
            identity_model: model
                .config_parameter_or("identity_model", "identity_gpu".to_string())?,
            held: AtomicI64::new(0),
            cancelled: AtomicI64::new(0),
        })?;
        model.report_memory_usage(&[BufferAttributes::new(
            MemoryType::Cpu,
            0,
            std::mem::size_of::<State>() as u64,
        )?])?;
        Ok(())
    }

    fn model_instance_initialize(instance: &ModelInstance) -> Result<(), Error> {
        instance.set_state(Buffers::default())?;
        instance.report_memory_usage(&[BufferAttributes::new(
            MemoryType::Cpu,
            0,
            std::mem::size_of::<Buffers>() as u64,
        )?])?;
        Ok(())
    }

    async fn model_instance_execute(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
        match ctx.state::<State>()?.mode {
            Mode::Mnist => execute_mnist(ctx, requests).await,
            Mode::Decoupled => {
                for request in requests {
                    execute_decoupled(request).await?;
                }
                Ok(())
            }
            Mode::StreamProxy => {
                for request in requests {
                    execute_stream_proxy(&ctx, request).await?;
                }
                Ok(())
            }
        }
    }
}

async fn execute_mnist(ctx: Context, requests: Vec<Request>) -> Result<(), Error> {
    let state = ctx.state::<State>()?;
    let mut held = Vec::new();
    let mut buffered = Vec::new();
    let mut pending = Vec::new();

    for request in requests {
        if let Some(trace) = request.view().trace()? {
            trace.report_activity("triton_ng.mnist.start")?;
        }
        if request.parameter("hold_until_cancelled")? == Some(Parameter::Bool(true)) {
            held.push(request);
            continue;
        }
        if let Some(Parameter::String(memory)) = request.parameter("memory")? {
            match memory.parse::<OutputMemory>() {
                Ok(memory) => buffered.push((request, memory)),
                Err(e) => request.respond_error(&e.to_string())?,
            }
            continue;
        }

        let image = request.get_input(IMAGE)?.as_bytes()?;
        pending.push((
            request,
            ctx.server()
                .infer_async(host_inference(&ctx, state, image)?),
        ));
    }

    for (request, inference) in pending {
        let digit = argmax(&inference.await?.output(LOGITS)?.as_fp32_vec()?)?;
        respond(&ctx, state, request, digit)?;
    }

    for (request, memory) in buffered {
        let image = request.get_input(IMAGE)?.as_bytes()?;
        match digit_with_buffers(&ctx, state, image, memory).await {
            Ok(digit) => respond(&ctx, state, request, digit)?,
            Err(e) => request.respond_error(&e.to_string())?,
        }
    }

    for request in held {
        hold_until_cancelled(state, request).await?;
    }

    Ok(())
}

struct StreamConfig {
    chunks: u64,
    early_release: bool,
    drop_factory: bool,
    drop_response: bool,
    delay: Duration,
    direct_final: bool,
}

impl StreamConfig {
    fn parse(values: &[f32]) -> Result<Self, Error> {
        if values.len() != 6 {
            return Err(format!("CONFIG expects 6 values, got {}", values.len()).into());
        }
        Ok(Self {
            chunks: stream_number(values[0], "chunks")?,
            early_release: stream_flag(values[1], "early_release")?,
            drop_factory: stream_flag(values[2], "drop_factory")?,
            drop_response: stream_flag(values[3], "drop_response")?,
            delay: Duration::from_millis(stream_number(values[4], "delay")?),
            direct_final: stream_flag(values[5], "direct_final")?,
        })
    }
}

fn stream_number(value: f32, name: &str) -> Result<u64, Error> {
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value > 1_000_000.0 {
        return Err(format!("{name} must be an integer between 0 and 1000000").into());
    }
    Ok(value as u64)
}

fn stream_flag(value: f32, name: &str) -> Result<bool, Error> {
    match value {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(format!("{name} must be 0 or 1").into()),
    }
}

async fn execute_decoupled(request: Request) -> Result<(), Error> {
    let config = StreamConfig::parse(&request.get_input("CONFIG")?.as_fp32_vec()?)?;
    if config.direct_final {
        let mut response = Response::new(request)?;
        response
            .create_output("OUTPUT", DataType::Fp32, &[1])?
            .write_fp32_vec(&[0.0])?;
        response.send()?;
        return Ok(());
    }

    let mut factory = ResponseFactory::new(request)?;
    if config.early_release {
        factory.release_request()?;
    }

    for index in 0..config.chunks {
        if factory.is_cancelled()? {
            CANCELLED_STREAMS.fetch_add(1, Ordering::SeqCst);
            factory.finish()?;
            return Ok(());
        }

        let mut response = factory.response()?;
        if config.drop_response {
            drop(response);
            factory.finish()?;
            return Ok(());
        }
        response
            .create_output("OUTPUT", DataType::Fp32, &[1])?
            .write_fp32_vec(&[index as f32])?;
        response.send()?;
        if !config.delay.is_zero() {
            tokio::time::sleep(config.delay).await;
        }
    }

    if config.drop_factory {
        drop(factory);
        return Ok(());
    }
    factory.finish()?;
    Ok(())
}

struct StreamReply {
    values: Vec<f32>,
    responses: i64,
    final_responses: i64,
    cancelled_streams: i64,
}

async fn execute_stream_proxy(ctx: &Context, request: Request) -> Result<(), Error> {
    let result = match prepare_stream(ctx, &request) {
        Ok((operation, inference)) => stream_reply(ctx, operation, inference).await,
        Err(error) => Err(error),
    };
    match result {
        Ok(reply) => respond_stream(request, reply),
        Err(error) => {
            request.respond_error(&error.to_string())?;
            Ok(())
        }
    }
}

fn prepare_stream(ctx: &Context, request: &Request) -> Result<(u64, InferenceRequest), Error> {
    let operation_values = request.get_input("OPERATION")?.as_fp32_vec()?;
    if operation_values.len() != 1 {
        return Err("OPERATION expects one value".into());
    }
    let operation = stream_number(operation_values[0], "operation")?;
    let config = request.get_input("CONFIG")?.as_fp32_vec()?;
    StreamConfig::parse(&config)?;

    let state = ctx.state::<State>()?;
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    let config_bytes = config
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    inference.add_input_with_data("CONFIG", DataType::Fp32, &[6], config_bytes)?;
    if operation == 3 {
        inference.add_requested_output_into("OUTPUT", ctx.memory().ram(4)?)?;
    } else {
        inference.add_requested_output("OUTPUT")?;
    }
    Ok((operation, inference))
}

async fn stream_reply(
    ctx: &Context,
    operation: u64,
    inference: InferenceRequest,
) -> Result<StreamReply, Error> {
    match operation {
        0 => {
            let mut stream = ctx.server().infer_stream(inference);
            let mut values = Vec::new();
            let mut responses = 0;
            let mut final_responses = 0;
            while let Some(result) = stream.next().await {
                let result = result?;
                responses += 1;
                final_responses += i64::from(result.is_final);
                for output in &result.outputs {
                    values.extend(output.as_fp32_vec()?);
                }
            }
            Ok(StreamReply {
                values,
                responses,
                final_responses,
                cancelled_streams: 0,
            })
        }
        1 => {
            let result = ctx.server().infer_async(inference).await?;
            let mut values = Vec::new();
            for output in &result.outputs {
                values.extend(output.as_fp32_vec()?);
            }
            Ok(StreamReply {
                values,
                responses: result.outputs.len() as i64,
                final_responses: i64::from(result.is_final),
                cancelled_streams: 0,
            })
        }
        2 => {
            let cancelled_before = CANCELLED_STREAMS.load(Ordering::SeqCst);
            let mut stream = ctx.server().infer_stream(inference);
            let first = stream
                .next()
                .await
                .ok_or("stream ended before its first response")??;
            let mut values = Vec::new();
            for output in &first.outputs {
                values.extend(output.as_fp32_vec()?);
            }
            let final_responses = i64::from(first.is_final);
            drop(first);
            drop(stream);

            for _ in 0..10_000 {
                if CANCELLED_STREAMS.load(Ordering::SeqCst) > cancelled_before {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
            let cancelled_streams = CANCELLED_STREAMS.load(Ordering::SeqCst);
            if cancelled_streams == cancelled_before {
                return Err("downstream stream was not cancelled".into());
            }
            Ok(StreamReply {
                values,
                responses: 1,
                final_responses,
                cancelled_streams,
            })
        }
        3 => {
            let mut stream = ctx.server().infer_stream(inference);
            let error = stream
                .next()
                .await
                .ok_or("infer_stream accepted a preallocated output")?
                .err()
                .ok_or("infer_stream accepted a preallocated output")?;
            Err(error.to_string().into())
        }
        other => Err(format!("unknown stream operation {other}").into()),
    }
}

fn respond_stream(request: Request, reply: StreamReply) -> Result<(), Error> {
    let mut response = Response::new(request)?;
    response
        .create_output("OUTPUT", DataType::Fp32, &[reply.values.len() as i64])?
        .write_fp32_vec(&reply.values)?;
    response.set_parameter("responses", reply.responses)?;
    response.set_parameter("final_responses", reply.final_responses)?;
    response.set_parameter("cancelled_streams", reply.cancelled_streams)?;
    response.send()?;
    Ok(())
}

fn host_inference(ctx: &Context, state: &State, image: Vec<u8>) -> Result<InferenceRequest, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    inference.add_input_with_data(IMAGE, DataType::Fp32, &IMAGE_SHAPE, image)?;
    inference.add_requested_output(LOGITS)?;
    Ok(inference)
}

async fn digit_with_buffers(
    ctx: &Context,
    state: &State,
    image: Vec<u8>,
    memory: OutputMemory,
) -> Result<i32, Error> {
    let buffers = ctx.instance_state::<Buffers>()?;
    let mut inference = InferenceRequest::new(ctx.server(), &state.target_model, -1)?;
    if let OutputMemory::Ram = memory {
        let mut ram_image = ctx.memory().ram(image.len())?;
        ram_image.copy_from_slice(&image);
        inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, ram_image)?;
    } else {
        let mut pinned_image = ctx.memory().pinned(image.len())?;
        pinned_image.copy_from_slice(&image);
        inference.add_input_with_buffer(IMAGE, DataType::Fp32, &IMAGE_SHAPE, pinned_image)?;
    }

    match memory {
        OutputMemory::Ram => {
            inference
                .add_requested_output_into(LOGITS, buffers.ram(ctx.memory(), LOGITS_BYTES)?)?;
            let mut result = ctx.server().infer_async(inference).await?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            buffers.put_ram(result.take_ram(LOGITS)?)?;
            Ok(digit)
        }
        OutputMemory::Pinned => {
            inference
                .add_requested_output_into(LOGITS, buffers.pinned(ctx.memory(), LOGITS_BYTES)?)?;
            let mut result = ctx.server().infer_async(inference).await?;
            let digit = argmax(&result.output(LOGITS)?.as_fp32_vec()?)?;
            buffers.put_pinned(result.take_pinned(LOGITS)?)?;
            Ok(digit)
        }
        OutputMemory::Gpu => {
            let logits = buffers.gpu(ctx.memory(), ctx.device_id(), LOGITS_BYTES)?;
            inference.add_requested_output_into(LOGITS, logits)?;
            let mut result = ctx.server().infer_async(inference).await?;
            buffers.put_gpu(result.take_gpu(LOGITS)?)?;

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::GpuInput => {
            inference.add_requested_output_into(
                LOGITS,
                ctx.memory().gpu(ctx.device_id(), LOGITS_BYTES)?,
            )?;
            let mut result = ctx.server().infer_async(inference).await?;
            let logits = Arc::new(result.take_gpu(LOGITS)?);
            let copies = [
                identity_of(ctx, state, &logits)?,
                identity_of(ctx, state, &logits)?,
            ];
            drop(logits);

            let inference = host_inference(ctx, state, image)?;
            let host = ctx.server().infer_async(inference).await?;
            let expected = host.output(LOGITS)?.bytes()?;
            for copy in copies {
                if copy.await?.output(IDENTITY_OUTPUT)?.bytes()? != expected {
                    return Err("the GPU buffer read as an input differs from the logits".into());
                }
            }
            argmax(&host.output(LOGITS)?.as_fp32_vec()?)
        }
        OutputMemory::WrongSize => {
            inference.add_requested_output_into(LOGITS, ctx.memory().pinned(4)?)?;
            ctx.server().infer_async(inference).await?;
            Err("output written into a buffer of the wrong size".into())
        }
    }
}

fn identity_of(
    ctx: &Context,
    state: &State,
    logits: &Arc<GpuBuffer>,
) -> Result<InferenceFuture, Error> {
    let mut inference = InferenceRequest::new(ctx.server(), &state.identity_model, -1)?;
    inference.add_input_with_buffer(
        IDENTITY_INPUT,
        DataType::Fp32,
        &[1, 10],
        Arc::clone(logits),
    )?;
    inference.add_requested_output(IDENTITY_OUTPUT)?;
    Ok(ctx.server().infer_async(inference))
}

fn argmax(logits: &[f32]) -> Result<i32, Error> {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(digit, _)| digit as i32)
        .ok_or_else(|| "mnist returned no logits".into())
}

fn respond(ctx: &Context, state: &State, request: Request, digit: i32) -> Result<(), Error> {
    let buffers = ctx.instance_state::<Buffers>()?;
    let mut response = Response::new(request)?;
    response
        .create_output("DIGIT", DataType::Int32, &[1, 1])?
        .write_i32_vec(&[digit])?;
    response.set_parameter("held_requests", state.held.load(Ordering::SeqCst))?;
    response.set_parameter("cancelled_requests", state.cancelled.load(Ordering::SeqCst))?;
    response.set_parameter(
        "ram_buffers_allocated",
        buffers.ram_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "pinned_buffers_allocated",
        buffers.pinned_allocated.load(Ordering::SeqCst),
    )?;
    response.set_parameter(
        "gpu_buffers_allocated",
        buffers.gpu_allocated.load(Ordering::SeqCst),
    )?;
    response.send()?;
    Ok(())
}

async fn hold_until_cancelled(state: &State, request: Request) -> Result<(), Error> {
    state.held.fetch_add(1, Ordering::SeqCst);
    let result = respond_when_cancelled(request).await;
    if result.is_ok() {
        state.cancelled.fetch_add(1, Ordering::SeqCst);
    }
    state.held.fetch_sub(1, Ordering::SeqCst);
    result
}

async fn respond_when_cancelled(request: Request) -> Result<(), Error> {
    while !request.is_cancelled()? {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    request.respond_cancelled()?;
    Ok(())
}
