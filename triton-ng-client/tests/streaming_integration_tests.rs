mod common;

use anyhow::{Context, Result};
use common::config::Config;
use common::dto::{InferInput, InferResult};
use common::sut::Sut;
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
