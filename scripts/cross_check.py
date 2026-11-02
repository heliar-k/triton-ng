# 独立交叉核对：直接向 mnist_onnx 发全零输入取 logits 的 argmax，
# 它必须等于 Rust 自定义后端（经 BLS 转发）返回的 DIGIT。
import json
import sys
import urllib.request

URL = "http://localhost:8000/v2"
ZEROS = [0.0] * 784
SHAPE = [1, 1, 28, 28]


def infer(model, input_name, output_name):
    body = json.dumps(
        {
            "inputs": [
                {"name": input_name, "shape": SHAPE, "datatype": "FP32", "data": ZEROS}
            ],
            "outputs": [{"name": output_name}],
        }
    ).encode()
    request = urllib.request.Request(
        f"{URL}/models/{model}/infer",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())["outputs"][0]["data"]


logits = infer("mnist_onnx", "Input3", "Plus214_Output_0")
digit = infer("mnist", "Input3", "DIGIT")[0]
expected = max(range(len(logits)), key=lambda index: logits[index])

print(f"mnist_onnx argmax={expected} mnist DIGIT={digit}")
print("logits:", [round(value, 4) for value in logits])

if digit != expected:
    sys.exit(f"不一致：后端返回 {digit}，目标模型 logits 的 argmax 是 {expected}")
