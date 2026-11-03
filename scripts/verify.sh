#!/usr/bin/env bash
# 验证 triton-ng 在某个 Triton 镜像上的功能：构建后端、校验导出符号、起服务器、
# 跑整套测试、运行示例客户端并做独立交叉核对。细节见 docs/verification.md。
#
# 用法: scripts/verify.sh [镜像标签]     默认 26.09-py3
set -euo pipefail

IMAGE_TAG="${1:-26.09-py3}"
IMAGE="nvcr.io/nvidia/tritonserver:${IMAGE_TAG}"
CONTAINER="triton-ng-verify"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SO="${ROOT}/target/release/libtriton_custom_backend.so"
MODELS="${ROOT}/target/verify-repo"

cd "$ROOT"

fail() {
  echo "验证失败: $*" >&2
  exit 1
}

# 本机 rust 的 libstd 可能要求更高版本的 glibc，优先用隔离安装的 rustup
if [[ -x "${HOME}/.local/share/cargo/bin/cargo" ]]; then
  export RUSTUP_HOME="${RUSTUP_HOME:-${HOME}/.local/share/rustup}"
  export CARGO_HOME="${CARGO_HOME:-${HOME}/.local/share/cargo}"
  export PATH="${CARGO_HOME}/bin:${PATH}"
fi

# bindgen 需要 libclang；优先用 PyPI wheel 里自带的那份（不依赖新版 glibc），其次系统目录
# 和 Homebrew（Homebrew 的 llvm 需要 GLIBC 2.38，Ubuntu 22.04 加载不了）
detect_libclang() {
  local found dir
  found="$(find "${HOME}/.local/share/libclang-py" -maxdepth 7 -name 'libclang.so' -print 2>/dev/null | head -1)"
  if [[ -n "${found}" ]]; then
    dirname "${found}"
    return
  fi
  for dir in /usr/lib/llvm-*/lib /usr/lib/x86_64-linux-gnu /home/linuxbrew/.linuxbrew/lib; do
    if [[ -e "${dir}/libclang.so" ]]; then
      echo "${dir}"
      return
    fi
  done
}

if [[ -z "${LIBCLANG_PATH:-}" ]]; then
  LIBCLANG_PATH="$(detect_libclang)"
  [[ -n "${LIBCLANG_PATH}" ]] || fail "找不到 libclang，安装 clang，或 pip download libclang --no-deps -d <目录> 后解开"
  export LIBCLANG_PATH
fi

command -v cargo-nextest >/dev/null || fail "缺少 cargo-nextest"
command -v docker >/dev/null || fail "缺少 docker"
command -v python3 >/dev/null || fail "缺少 python3"

echo "== 工具链: $(cargo --version), libclang: ${LIBCLANG_PATH}"

echo "== 构建 =="
make build

echo "== 校验导出符号 =="
expected="$(printf '%s\n' \
  TRITONBACKEND_Finalize \
  TRITONBACKEND_Initialize \
  TRITONBACKEND_ModelFinalize \
  TRITONBACKEND_ModelInitialize \
  TRITONBACKEND_ModelInstanceExecute \
  TRITONBACKEND_ModelInstanceFinalize \
  TRITONBACKEND_ModelInstanceInitialize | sort)"
actual="$(nm -D --defined-only "${SO}" | awk '{print $3}' | grep '^TRITONBACKEND_' | sort)"
[[ "${actual}" == "${expected}" ]] || fail "导出符号与预期不符，实际为: $(echo "${actual}" | tr '\n' ' ')"

echo "== 组装模型仓库 =="
[[ -f models/mnist_onnx/1/model.onnx ]] || make download-model
for model in mnist mnist_onnx identity_gpu decoupled stream_proxy; do
  mkdir -p "${MODELS}/${model}/1"
  cp "verify/models/${model}/config.pbtxt" "${MODELS}/${model}/config.pbtxt"
done
cp models/mnist_onnx/1/model.onnx "${MODELS}/mnist_onnx/1/model.onnx"
mkdir -p "${ROOT}/target/traces"

echo "== 启动 Triton ${IMAGE_TAG} =="
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}"
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

cleanup() {
  local status=$?
  if ((status != 0)); then
    echo "== 服务器日志尾部 ==" >&2
    docker logs "${CONTAINER}" 2>&1 | tail -40 >&2 || true
    echo "容器 ${CONTAINER} 保留用于排查" >&2
  else
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# 客户端共享内存在 26.x 起默认关闭，靠镜像自己的 --help 判断该参数是否存在。
# tritonserver --help 退出码为 1，且输出可能走 stderr，所以整段抓下来再匹配。
shm_flag=()
help_text="$(docker run --rm --entrypoint tritonserver "${IMAGE}" --help 2>&1 || true)"
if grep -q -- '--allow-client-shm' <<<"${help_text}"; then
  shm_flag=(--allow-client-shm=true)
  echo "   该镜像支持 --allow-client-shm，测试时开启"
fi

docker run -d --name "${CONTAINER}" \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${MODELS}:/models" \
  -v "${SO}:/opt/tritonserver/backends/custom_backend/libtriton_custom_backend.so" \
  -v "${ROOT}/target/traces:/traces" \
  "${IMAGE}" tritonserver \
  --model-repository=/models \
  --model-control-mode=explicit \
  --log-verbose=1 \
  ${shm_flag[@]+"${shm_flag[@]}"} \
  --load-model=mnist \
  --load-model=mnist_onnx \
  --load-model=identity_gpu \
  --load-model=decoupled \
  --load-model=stream_proxy \
  --trace-config triton,file=/traces/trace.json \
  --trace-config triton,log-frequency=1 \
  --trace-config rate=1 \
  --trace-config level=TIMESTAMPS >/dev/null

echo "== 等待服务器就绪 =="
for _ in $(seq 1 60); do
  if curl -fsS -o /dev/null http://localhost:8000/v2/health/ready; then
    break
  fi
  sleep 2
done
curl -fsS -o /dev/null http://localhost:8000/v2/health/ready || fail "服务器在 120 秒内未就绪"

echo "== 整套测试 =="
make tests

echo "== 示例客户端 =="
cargo run --manifest-path=example/app/Cargo.toml --release | tee "${ROOT}/target/verify-app.log"

echo "== 独立交叉核对 =="
python3 scripts/cross_check.py

echo "== 通过: ${IMAGE_TAG} =="
