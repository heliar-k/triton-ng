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
        None,
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

#[tokio::test]
async fn custom_backend_reports_request_statistics() -> Result<()> {
    infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;

    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics_for("decoupled").await?.into_proto();
    let inference = stats.inference_stats.context("inference_stats missing")?;
    let success = inference.success.context("success statistics missing")?;
    assert!(success.count > 0);

    for duration in [
        inference.queue,
        inference.compute_input,
        inference.compute_infer,
        inference.compute_output,
    ] {
        let duration = duration.context("request duration statistics missing")?;
        assert_eq!(duration.count, success.count);
        assert!(duration.ns < success.count * 60_000_000_000);
    }
    Ok(())
}

#[tokio::test]
async fn custom_backend_reports_statistics_for_every_response() -> Result<()> {
    infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;

    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics_for("decoupled").await?.into_proto();
    for index in ["0", "1", "2"] {
        let response = stats
            .response_stats
            .get(index)
            .with_context(|| format!("response statistics {index} not found"))?;
        let success = response.success.context("response success count missing")?;
        assert!(success.count > 0);
        assert!(
            response
                .compute_infer
                .context("compute_infer missing")?
                .count
                >= success.count
        );
        assert!(
            response
                .compute_output
                .context("compute_output missing")?
                .count
                >= success.count
        );
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
async fn custom_backend_reports_prometheus_statistics() -> Result<()> {
    infer_stream([3.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;
    let output = Command::new("curl")
        .args(["-fsS", "http://localhost:8002/metrics"])
        .output()?;
    if !output.status.success() {
        bail!("failed to read Prometheus metrics: {output:?}");
    }
    let metrics = String::from_utf8(output.stdout)?;

    for name in [
        "nv_inference_count",
        "nv_inference_request_success",
        "nv_inference_request_duration_us",
        "nv_inference_queue_duration_us",
        "nv_inference_compute_input_duration_us",
        "nv_inference_compute_infer_duration_us",
        "nv_inference_compute_output_duration_us",
    ] {
        let line = metrics
            .lines()
            .find(|line| {
                line.starts_with(name)
                    && line.contains("model=\"decoupled\"")
                    && line.contains("version=\"1\"")
            })
            .with_context(|| format!("Prometheus metric {name} not found"))?;
        let value = line
            .split_whitespace()
            .last()
            .context("Prometheus metric has no value")?
            .parse::<f64>()?;
        assert!(value.is_finite() && value >= 0.0);
        if name == "nv_inference_count" || name == "nv_inference_request_success" {
            assert!(value > 0.0);
        }
    }
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
        if trace_files_contain(&trace_dir, &request_id, "triton_ng.mnist.start")? {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    bail!("custom trace activity not found")
}

fn trace_files_contain(directory: &Path, request_id: &str, activity: &str) -> Result<bool> {
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("trace.json"))
        {
            let trace = std::fs::read_to_string(path)?;
            if trace.contains(request_id) && trace.contains(activity) {
                return Ok(true);
            }
        }
    }
    Ok(false)
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
    let result = infer_stream([3.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.0).await?;
    assert_eq!(values(&result)?, [0.0, 1.0, 2.0]);
    assert_eq!(parameter(&result, "responses")?, 3);
    Ok(())
}

#[tokio::test]
async fn dropping_unsent_response_produces_an_error_response() -> Result<()> {
    let error = infer_stream([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0.0)
        .await
        .err()
        .context("dropped response unexpectedly succeeded")?;
    assert!(
        format!("{error:#}").contains("backend did not send the response"),
        "{error:#}"
    );

    let config = Config::load()?;
    let sut = Sut::new(&config.triton_url).await?;
    let stats = sut.model_statistics_for("decoupled").await?.into_proto();
    let failure = stats
        .response_stats
        .get("0")
        .context("error response statistics not found")?
        .fail
        .context("error response failure count missing")?;
    assert!(failure.count > 0);
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
