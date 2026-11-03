use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use triton_ng_client::{
    InferInput, InferOptions, InferResponse, OutputData, Parameter, TritonClient,
    TritonClientConfig,
};

const ECHO_OUTPUTS: [&str; 9] = [
    "FP16", "BF16", "FP64", "INT8", "UINT8", "UINT16", "UINT32", "BOOL", "METADATA",
];

#[tokio::test]
async fn typed_tensors_round_trip_through_the_custom_backend() -> Result<()> {
    let client = client().await?;
    let fp16: Vec<u8> = [0x3c00u16, 0xc000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let bf16: Vec<u8> = [0x3f80u16, 0x4000]
        .into_iter()
        .flat_map(u16::to_le_bytes)
        .collect();
    let inputs = vec![
        InferInput::fp16("FP16", vec![2], fp16.clone()),
        InferInput::bf16("BF16", vec![2], bf16.clone()),
        InferInput::fp64("FP64", vec![2], vec![1.25, -2.5]),
        InferInput::int8("INT8", vec![3], vec![-128, 0, 127]),
        InferInput::uint8("UINT8", vec![3], vec![0, 128, 255]),
        InferInput::uint16("UINT16", vec![2], vec![0, 65535]),
        InferInput::uint32("UINT32", vec![2], vec![0, 4_000_000_000]),
        InferInput::bool("BOOL", vec![3], vec![true, false, true]),
    ];
    let before = response_compute_infer(&client).await?;
    let response = client
        .infer(
            "tensor_echo",
            None,
            inputs,
            ECHO_OUTPUTS,
            InferOptions {
                parameters: HashMap::from([("output_delay".to_string(), Parameter::Bool(true))]),
                ..Default::default()
            },
        )
        .await?;

    let expected = [
        ("FP16", OutputData::Fp16(fp16)),
        ("BF16", OutputData::Bf16(bf16)),
        ("FP64", OutputData::Fp64(vec![1.25, -2.5])),
        ("INT8", OutputData::Int8(vec![-128, 0, 127])),
        ("UINT8", OutputData::Uint8(vec![0, 128, 255])),
        ("UINT16", OutputData::Uint16(vec![0, 65535])),
        ("UINT32", OutputData::Uint32(vec![0, 4_000_000_000])),
        ("BOOL", OutputData::Bool(vec![true, false, true])),
    ];
    for (name, expected) in expected {
        let output = response
            .outputs
            .iter()
            .find(|output| output.name == name)
            .with_context(|| format!("output {name} not found"))?;
        assert_eq!(output.data, expected, "output {name}");
    }
    let after = response_compute_infer(&client).await?;
    assert_eq!(after.0, before.0 + 1);
    assert!(after.1 >= before.1 + 25_000_000);
    Ok(())
}

#[tokio::test]
async fn request_and_server_metadata_are_readable() -> Result<()> {
    let client = client().await?;
    let response = client
        .infer(
            "tensor_echo",
            None,
            metadata_inputs(),
            ["METADATA"],
            InferOptions {
                id: Some("direct-request".to_string()),
                timeout: Some(Duration::from_micros(5_000_000)),
                parameters: HashMap::from([
                    ("bool".to_string(), Parameter::Bool(true)),
                    ("int".to_string(), Parameter::Int(-7)),
                    ("double".to_string(), Parameter::Double(0.5)),
                    (
                        "string".to_string(),
                        Parameter::String("client".to_string()),
                    ),
                ]),
            },
        )
        .await?;
    let metadata = metadata(&response)?;

    assert_eq!(metadata["id"], "direct-request");
    assert_eq!(metadata["correlation_id"], 0);
    assert!(metadata["correlation_id_string"].is_null());
    assert_eq!(metadata["sequence_start"], false);
    assert_eq!(metadata["sequence_end"], false);
    assert_eq!(metadata["timeout"], 0);
    assert_eq!(metadata["bool_parameter"], true);
    assert_eq!(metadata["int_parameter"], -7);
    assert_eq!(metadata["double_parameter"], 0.5);
    assert_eq!(metadata["string_parameter"], "client");
    assert_eq!(metadata["host_policy"], "cpu");
    assert_eq!(metadata["host_policy_json"], serde_json::json!({"cpu": {}}));
    assert!(
        metadata["parallel_instance_loads"]
            .as_u64()
            .context("parallel_instance_loads is not an integer")?
            >= 2
    );
    assert_eq!(metadata["server_reads_rejected_before_ready"], true);
    assert_eq!(metadata["trace_available"], true);
    assert_eq!(metadata["server_config"], true);
    assert_eq!(metadata["server_statistics"], true);
    assert_eq!(metadata["server_index"], true);
    Ok(())
}

#[tokio::test]
async fn bls_request_forwards_numeric_metadata_and_typed_data() -> Result<()> {
    let metadata = infer_proxy("numeric").await?;
    assert_eq!(metadata["correlation_id"], 42);
    assert!(metadata["correlation_id_string"].is_null());
    assert_bls_metadata(&metadata, true, false);
    Ok(())
}

#[tokio::test]
async fn bls_request_forwards_string_correlation_id() -> Result<()> {
    let metadata = infer_proxy("string").await?;
    assert!(metadata["correlation_id"].is_null());
    assert_eq!(metadata["correlation_id_string"], "sequence-alpha");
    assert_bls_metadata(&metadata, false, true);
    Ok(())
}

#[tokio::test]
async fn bls_request_accepts_priority_above_the_model_limit() -> Result<()> {
    let metadata = infer_proxy("overflow").await?;
    assert_eq!(metadata["correlation_id"], 43);
    assert_bls_metadata(&metadata, true, true);
    Ok(())
}

#[tokio::test]
async fn bls_request_returns_every_typed_input() -> Result<()> {
    let response = infer_proxy_response("numeric").await?;
    let expected = [
        ("FP16", OutputData::Fp16(0x3c00u16.to_le_bytes().to_vec())),
        ("BF16", OutputData::Bf16(0x3f80u16.to_le_bytes().to_vec())),
        ("FP64", OutputData::Fp64(vec![1.25])),
        ("INT8", OutputData::Int8(vec![-1])),
        ("UINT8", OutputData::Uint8(vec![1])),
        ("UINT16", OutputData::Uint16(vec![1])),
        ("UINT32", OutputData::Uint32(vec![1])),
        ("BOOL", OutputData::Bool(vec![true])),
    ];
    for (name, expected) in expected {
        let output = response
            .outputs
            .iter()
            .find(|output| output.name == name)
            .with_context(|| format!("output {name} not found"))?;
        assert_eq!(output.data, expected, "output {name}");
    }
    Ok(())
}

#[tokio::test]
async fn unsampled_request_has_no_trace_view() -> Result<()> {
    let client = client().await?;
    let settings = client
        .trace_settings(
            HashMap::from([("trace_level".to_string(), vec!["OFF".to_string()])]),
            Some("tensor_echo"),
            None,
        )
        .await?;
    let response = client
        .infer(
            "tensor_echo",
            None,
            metadata_inputs(),
            ["METADATA"],
            InferOptions::default(),
        )
        .await;
    client
        .trace_settings(
            HashMap::from([("trace_level".to_string(), vec!["TIMESTAMPS".to_string()])]),
            Some("tensor_echo"),
            None,
        )
        .await?;
    assert_eq!(
        settings.get("trace_level").map(Vec::as_slice),
        Some(&["OFF".to_string()][..]),
        "{settings:?}"
    );
    let metadata = metadata(&response?)?;
    assert_eq!(metadata["trace_available"], false);
    Ok(())
}

#[tokio::test]
async fn preferred_instance_group_is_used_when_config_omits_it() -> Result<()> {
    let client = client().await?;
    let model_config = client
        .model_config("tensor_echo", None, None)
        .await?
        .context("tensor_echo config not found")?
        .into_proto();
    assert_eq!(model_config.instance_group.len(), 1);
    assert_eq!(model_config.instance_group[0].count, 2);
    assert_eq!(model_config.instance_group[0].kind, 2);
    Ok(())
}

async fn client() -> Result<TritonClient> {
    let config: Value = serde_json::from_str(include_str!("config.json"))?;
    let url = config["triton_url"]
        .as_str()
        .context("triton_url missing from test config")?;
    Ok(TritonClient::new(TritonClientConfig::new(url)).await?)
}

async fn response_compute_infer(client: &TritonClient) -> Result<(u64, u64)> {
    let stats = client
        .model_statistics(Some("tensor_echo"), None, None)
        .await?
        .into_iter()
        .find(|stats| stats.name() == "tensor_echo")
        .context("tensor_echo statistics not found")?
        .into_proto();
    let duration = stats
        .response_stats
        .get("0")
        .and_then(|stats| stats.compute_infer)
        .map_or((0, 0), |duration| (duration.count, duration.ns));
    Ok(duration)
}

async fn infer_proxy(kind: &str) -> Result<Value> {
    metadata(&infer_proxy_response(kind).await?)
}

async fn infer_proxy_response(kind: &str) -> Result<InferResponse> {
    let client = client().await?;
    Ok(client
        .infer(
            "tensor_echo",
            None,
            metadata_inputs(),
            ECHO_OUTPUTS,
            InferOptions {
                parameters: HashMap::from([(
                    "proxy".to_string(),
                    Parameter::String(kind.to_string()),
                )]),
                ..Default::default()
            },
        )
        .await?)
}

fn assert_bls_metadata(metadata: &Value, sequence_start: bool, sequence_end: bool) {
    assert_eq!(metadata["id"], "");
    assert_eq!(metadata["sequence_start"], sequence_start);
    assert_eq!(metadata["sequence_end"], sequence_end);
    assert_eq!(metadata["timeout"], 5_000_000);
    assert_eq!(metadata["bool_parameter"], true);
    assert_eq!(metadata["int_parameter"], 99);
    assert_eq!(metadata["double_parameter"], 2.5);
    assert_eq!(metadata["string_parameter"], "downstream");
}

fn metadata(response: &InferResponse) -> Result<Value> {
    let output = response
        .outputs
        .iter()
        .find(|output| output.name == "METADATA")
        .context("METADATA output not found")?;
    let OutputData::Bytes(values) = &output.data else {
        anyhow::bail!("METADATA has unexpected datatype {:?}", output.datatype)
    };
    let value = values.first().context("METADATA is empty")?;
    Ok(serde_json::from_slice(value)?)
}

fn metadata_inputs() -> Vec<InferInput> {
    vec![
        InferInput::fp16("FP16", vec![1], 0x3c00u16.to_le_bytes().to_vec()),
        InferInput::bf16("BF16", vec![1], 0x3f80u16.to_le_bytes().to_vec()),
        InferInput::fp64("FP64", vec![1], vec![1.25]),
        InferInput::int8("INT8", vec![1], vec![-1]),
        InferInput::uint8("UINT8", vec![1], vec![1]),
        InferInput::uint16("UINT16", vec![1], vec![1]),
        InferInput::uint32("UINT32", vec![1], vec![1]),
        InferInput::bool("BOOL", vec![1], vec![true]),
    ]
}
