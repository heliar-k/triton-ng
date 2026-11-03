mod common;

use anyhow::{Context, Result, bail};
use common::config::Config;
use common::dto::{InferInput, InferResult};
use common::sut::Sut;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use triton_ng_client::Parameter;

async fn infer_stream(config: [f32; 6], operation: f32) -> Result<InferResult> {
    infer_stream_with_id(config, operation, None).await
}

async fn infer_stream_with_id(
    config: [f32; 6],
    operation: f32,
    id: Option<String>,
) -> Result<InferResult> {
    let test_config = Config::load()?;
    let sut = Sut::new(&test_config.triton_url).await?;
    sut.infer(
        &test_config.stream_proxy_model,
        None,
        vec![
            InferInput {
                name: "CONFIG".to_string(),
                shape: vec![6],
                data: config.to_vec(),
            },
            InferInput {
                name: "OPERATION".to_string(),
                shape: vec![1],
                data: vec![operation],
            },
        ],
        &["OUTPUT"],
        id,
    )
    .await
}

fn values(result: &InferResult) -> Result<&[f32]> {
    let output = result
        .outputs
        .iter()
        .find(|output| output.name == "OUTPUT")
        .context("OUTPUT not found")?;
    Ok(&output.data_f32)
}

fn parameter(result: &InferResult, name: &str) -> Result<i64> {
    match result.parameters.get(name) {
        Some(Parameter::Int(value)) => Ok(*value),
        other => anyhow::bail!("response parameter {name} is {other:?}"),
    }
}

#[derive(Clone, Copy, Default)]
struct DurationStats {
    count: u64,
    ns: u64,
}

macro_rules! duration_stats {
    ($value:expr) => {
        $value.map_or(DurationStats { count: 0, ns: 0 }, |value| DurationStats {
            count: value.count,
            ns: value.ns,
        })
    };
}

struct RequestStats {
    success: DurationStats,
    fail: DurationStats,
    queue: DurationStats,
    compute_input: DurationStats,
    compute_infer: DurationStats,
    compute_output: DurationStats,
}

#[derive(Default)]
struct ResponseStats {
    compute_infer: DurationStats,
    compute_output: DurationStats,
    success: DurationStats,
    fail: DurationStats,
    empty: DurationStats,
}

async fn request_stats(sut: &Sut, model: &str) -> Result<RequestStats> {
    let stats = sut.model_statistics_for(model).await?.into_proto();
    let stats = stats.inference_stats.context("inference_stats missing")?;
    Ok(RequestStats {
        success: duration_stats!(stats.success),
        fail: duration_stats!(stats.fail),
        queue: duration_stats!(stats.queue),
        compute_input: duration_stats!(stats.compute_input),
        compute_infer: duration_stats!(stats.compute_infer),
        compute_output: duration_stats!(stats.compute_output),
    })
}

async fn response_stats(sut: &Sut, model: &str, index: &str) -> Result<ResponseStats> {
    let stats = sut.model_statistics_for(model).await?.into_proto();
    let Some(stats) = stats.response_stats.get(index) else {
        return Ok(ResponseStats::default());
    };
    Ok(ResponseStats {
        compute_infer: duration_stats!(stats.compute_infer),
        compute_output: duration_stats!(stats.compute_output),
        success: duration_stats!(stats.success),
        fail: duration_stats!(stats.fail),
        empty: duration_stats!(stats.empty_response),
    })
}

fn prometheus_metrics() -> Result<String> {
    let output = Command::new("curl")
        .args(["-fsS", "http://localhost:8002/metrics"])
        .output()?;
    if !output.status.success() {
        bail!("failed to read Prometheus metrics: {output:?}");
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn prometheus_value(metrics: &str, name: &str) -> Result<f64> {
    let mut found = false;
    let mut total = 0.0;
    for line in metrics.lines().filter(|line| {
        line.starts_with(name)
            && line.contains("model=\"decoupled\"")
            && line.contains("version=\"1\"")
    }) {
        found = true;
        let value = line
            .split_whitespace()
            .last()
            .context("Prometheus metric has no value")?
            .parse::<f64>()?;
        assert!(value.is_finite() && value >= 0.0);
        total += value;
    }
    if !found {
        bail!("Prometheus metric {name} not found");
    }
    Ok(total)
}

#[tokio::test]
async fn custom_backend_reports_request_statistics() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = request_stats(&sut, "decoupled").await?;
    infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;
    let after = request_stats(&sut, "decoupled").await?;

    assert_eq!(after.success.count, before.success.count + 1);
    assert_eq!(after.fail.count, before.fail.count);
    for (before, after) in [
        (before.queue, after.queue),
        (before.compute_input, after.compute_input),
        (before.compute_infer, after.compute_infer),
        (before.compute_output, after.compute_output),
    ] {
        assert_eq!(after.count, before.count + 1);
        assert!(after.ns > before.ns);
        assert!(after.ns - before.ns < 60_000_000_000);
    }
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_statistics_for_every_response() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let mut before = Vec::new();
    for index in ["0", "1", "2"] {
        before.push(response_stats(&sut, "decoupled", index).await?);
    }
    infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;

    for (index, before) in before.into_iter().enumerate() {
        let after = response_stats(&sut, "decoupled", &index.to_string()).await?;
        assert_eq!(after.success.count, before.success.count + 1);
        assert_eq!(after.fail.count, before.fail.count);
        assert_eq!(after.compute_infer.count, before.compute_infer.count + 1);
        assert_eq!(after.compute_output.count, before.compute_output.count + 1);
        assert!(after.compute_infer.ns >= before.compute_infer.ns);
        assert!(after.compute_output.ns >= before.compute_output.ns);
    }
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_memory_usage() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics_for("decoupled").await?.into_proto();
    let cpu = stats
        .memory_usage
        .iter()
        .find(|usage| usage.r#type == "CPU" && usage.id == 0)
        .context("CPU memory usage not found")?;
    assert!(cpu.byte_size > 0);
    Ok(())
}

#[tokio::test]
async fn memory_reports_replace_and_clear_previous_entries() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics_for("tensor_echo").await?.into_proto();
    assert_eq!(stats.memory_usage.len(), 1);
    assert_eq!(stats.memory_usage[0].r#type, "CPU");
    assert_eq!(stats.memory_usage[0].id, 0);
    assert_eq!(stats.memory_usage[0].byte_size, 4096);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_prometheus_statistics() -> Result<()> {
    let before = prometheus_metrics()?;
    infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;
    let after = prometheus_metrics()?;

    for name in [
        "nv_inference_count",
        "nv_inference_request_success",
        "nv_inference_request_duration_us",
        "nv_inference_queue_duration_us",
        "nv_inference_compute_input_duration_us",
        "nv_inference_compute_infer_duration_us",
        "nv_inference_compute_output_duration_us",
    ] {
        let before = prometheus_value(&before, name)?;
        let after = prometheus_value(&after, name)?;
        if name == "nv_inference_count" || name == "nv_inference_request_success" {
            assert!(after > before, "Prometheus metric {name} did not increase");
        } else {
            // 微秒精度下计算耗时可舍入为 0，纳秒增量由 model statistics 测试检查。
            assert!(after >= before, "Prometheus metric {name} decreased");
        }
    }

    let failure_before = prometheus_value(&after, "nv_inference_request_failure")?;
    infer_stream([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.0)
        .await
        .err()
        .context("dropped response unexpectedly succeeded")?;
    let failure_after = prometheus_value(&prometheus_metrics()?, "nv_inference_request_failure")?;
    assert!(failure_after > failure_before);
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_trace_activity() -> Result<()> {
    let config = Config::load()?;
    let trace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/traces");
    std::fs::create_dir_all(&trace_dir)?;
    let request_id = format!(
        "trace-activity-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );

    let sut = Sut::new(&config.triton_url).await?;
    let output_name = config.backend_output_name.clone();
    sut.infer(
        &config.backend_model,
        None,
        vec![config.mnist_blank_input()],
        &[&output_name],
        Some(request_id.clone()),
    )
    .await?;

    for _ in 0..100 {
        if trace_request_matches(&trace_dir, &request_id, Some("triton_ng.mnist.start"))? {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    bail!("custom trace activity not found")
}

fn trace_request_matches(
    directory: &Path,
    request_id: &str,
    activity: Option<&str>,
) -> Result<bool> {
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if !path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("trace.json"))
        {
            continue;
        }
        let Ok(events) =
            serde_json::from_str::<Vec<serde_json::Value>>(&std::fs::read_to_string(path)?)
        else {
            // 文件仍在写入时，下一轮继续等待完整 JSON。
            continue;
        };
        let Some(id) = events
            .iter()
            .find(|event| event["request_id"] == request_id)
            .map(|event| event["id"].clone())
        else {
            continue;
        };
        let timestamps: std::collections::HashMap<String, u64> = events
            .iter()
            .filter(|event| event["id"] == id)
            .filter_map(|event| event["timestamps"].as_array())
            .flatten()
            .filter_map(|timestamp| {
                Some((
                    timestamp["name"].as_str()?.to_string(),
                    timestamp["ns"].as_u64()?,
                ))
            })
            .collect();
        if activity.is_some_and(|name| !timestamps.contains_key(name)) {
            continue;
        }
        let mut previous = 0;
        let mut complete = true;
        for name in [
            "COMPUTE_START",
            "COMPUTE_INPUT_END",
            "COMPUTE_OUTPUT_START",
            "COMPUTE_END",
            "REQUEST_END",
        ] {
            let Some(timestamp) = timestamps.get(name) else {
                complete = false;
                break;
            };
            if *timestamp < previous {
                complete = false;
                break;
            }
            previous = *timestamp;
        }
        if complete {
            return Ok(true);
        }
    }
    Ok(false)
}

#[tokio::test]
async fn failed_request_trace_has_ordered_backend_timestamps() -> Result<()> {
    let trace_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/traces");
    std::fs::create_dir_all(&trace_dir)?;
    let request_id = format!(
        "trace-failure-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    infer_stream_with_id(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        0.0,
        Some(request_id.clone()),
    )
    .await
    .err()
    .context("dropped response unexpectedly succeeded")?;

    for _ in 0..100 {
        if trace_request_matches(&trace_dir, &request_id, None)? {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    bail!("failed request trace is missing ordered backend timestamps")
}

#[tokio::test]
async fn infer_stream_receives_every_decoupled_response() -> Result<()> {
    let result = infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;
    assert_eq!(values(&result)?, [0.0, 1.0, 2.0]);
    assert_eq!(parameter(&result, "responses")?, 3);
    assert_eq!(parameter(&result, "final_responses")?, 0);
    Ok(())
}

#[tokio::test]
async fn responses_continue_after_early_request_release() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = request_stats(&sut, "decoupled").await?;
    let result = infer_stream([3.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;
    assert_eq!(values(&result)?, [0.0, 1.0, 2.0]);
    assert_eq!(parameter(&result, "responses")?, 3);

    let after = request_stats(&sut, "decoupled").await?;
    assert_eq!(after.fail.count, before.fail.count + 1);
    assert_eq!(after.success.count, before.success.count);
    assert_eq!(after.queue.count, before.queue.count);
    assert_eq!(after.compute_input.count, before.compute_input.count);
    assert_eq!(after.compute_infer.count, before.compute_infer.count);
    assert_eq!(after.compute_output.count, before.compute_output.count);
    Ok(())
}

#[tokio::test]
async fn dropping_unsent_response_produces_an_error_response() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let request_before = request_stats(&sut, "decoupled").await?;
    let response_before = response_stats(&sut, "decoupled", "0").await?;
    let error = infer_stream([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.0)
        .await
        .err()
        .context("dropped response unexpectedly succeeded")?;
    assert!(
        format!("{error:#}").contains("backend did not send the response"),
        "{error:#}"
    );

    let request_after = request_stats(&sut, "decoupled").await?;
    let response_after = response_stats(&sut, "decoupled", "0").await?;
    assert_eq!(request_after.fail.count, request_before.fail.count + 1);
    assert_eq!(request_after.success.count, request_before.success.count);
    assert_eq!(response_after.fail.count, response_before.fail.count + 1);
    assert_eq!(response_after.success.count, response_before.success.count);
    Ok(())
}

#[tokio::test]
async fn dropping_owned_response_produces_an_error_response() -> Result<()> {
    let error = infer_stream([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], 0.0)
        .await
        .err()
        .context("dropped owned response unexpectedly succeeded")?;
    assert!(
        format!("{error:#}").contains("backend did not send the response"),
        "{error:#}"
    );
    Ok(())
}

#[tokio::test]
async fn dropping_request_produces_an_error_response() -> Result<()> {
    let error = infer_stream([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], 0.0)
        .await
        .err()
        .context("dropped request unexpectedly succeeded")?;
    assert!(
        format!("{error:#}").contains("backend did not produce a response for the request"),
        "{error:#}"
    );
    Ok(())
}

#[tokio::test]
async fn successful_empty_response_has_no_output_duration() -> Result<()> {
    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let before = response_stats(&sut, "decoupled", "0").await?;
    let result = infer_stream([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0).await?;
    assert!(values(&result)?.is_empty());
    assert_eq!(parameter(&result, "responses")?, 1);
    assert_eq!(parameter(&result, "final_responses")?, 1);

    let after = response_stats(&sut, "decoupled", "0").await?;
    assert_eq!(after.success.count, before.success.count);
    assert_eq!(after.empty.count, before.empty.count + 1);
    assert_eq!(after.compute_output.count, before.compute_output.count);
    Ok(())
}

#[tokio::test]
async fn dropping_unfinished_factory_produces_an_error_response() -> Result<()> {
    let error = infer_stream([2.0, 0.0, 1.0, 0.0, 0.0, 0.0], 0.0)
        .await
        .err()
        .context("dropped factory unexpectedly succeeded")?;
    assert!(
        format!("{error:#}").contains("backend did not finish the response stream"),
        "{error:#}"
    );
    Ok(())
}

#[tokio::test]
async fn infer_async_aggregates_decoupled_responses() -> Result<()> {
    let result = infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0).await?;
    assert_eq!(values(&result)?, [0.0, 1.0, 2.0]);
    assert_eq!(parameter(&result, "responses")?, 3);
    assert_eq!(parameter(&result, "final_responses")?, 1);
    Ok(())
}

#[tokio::test]
async fn final_response_carries_final_flag() -> Result<()> {
    let result = infer_stream([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 0.0).await?;
    assert_eq!(parameter(&result, "responses")?, 1);
    assert_eq!(parameter(&result, "final_responses")?, 1);
    Ok(())
}

#[tokio::test]
async fn dropping_inference_stream_cancels_downstream_after_early_release() -> Result<()> {
    let result = infer_stream([1000.0, 1.0, 0.0, 0.0, 5.0, 0.0], 2.0).await?;
    assert_eq!(values(&result)?, [0.0]);
    assert!(parameter(&result, "cancelled_streams")? > 0);
    Ok(())
}

#[tokio::test]
async fn infer_stream_rejects_preallocated_output_before_submission() -> Result<()> {
    let error = infer_stream([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3.0)
        .await
        .err()
        .context("preallocated output unexpectedly succeeded")?;
    assert!(
        format!("{error:#}").contains("preallocated output buffers are not supported"),
        "{error:#}"
    );
    Ok(())
}
