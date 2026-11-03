use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use triton_ng_client::{
    InferInput, InferOptions, InferResponse, OutputData, Parameter, TritonClient,
    TritonClientConfig,
};

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
    let output_names = [
        "FP16", "BF16", "FP64", "INT8", "UINT8", "UINT16", "UINT32", "BOOL", "METADATA",
    ];

    let response = client
        .infer(
            "tensor_echo",
            None,
            inputs,
            output_names,
            InferOptions::default(),
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
    assert_bls_metadata(&metadata);
    Ok(())
}

#[tokio::test]
async fn bls_request_forwards_string_correlation_id() -> Result<()> {
    let metadata = infer_proxy("string").await?;
    assert!(metadata["correlation_id"].is_null());
    assert_eq!(metadata["correlation_id_string"], "sequence-alpha");
    assert_bls_metadata(&metadata);
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

async fn infer_proxy(kind: &str) -> Result<Value> {
    let client = client().await?;
    let response = client
        .infer(
            "tensor_echo",
            None,
            metadata_inputs(),
            ["METADATA"],
            InferOptions {
                parameters: HashMap::from([(
                    "proxy".to_string(),
                    Parameter::String(kind.to_string()),
                )]),
                ..Default::default()
            },
        )
        .await?;
    metadata(&response)
}

fn assert_bls_metadata(metadata: &Value) {
    assert_eq!(metadata["id"], "");
    assert_eq!(metadata["sequence_start"], true);
    assert_eq!(metadata["sequence_end"], true);
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
