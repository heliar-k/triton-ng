setup-protoc:
	sudo dnf install protobuf-compiler
	protoc --version

docker-env-up:
	docker compose up -d

docker-env-down:
	docker compose down

docker-env-clean:
	docker compose down -v

docker-logs:
	docker logs triton

format:
	cargo sort --workspace
	cargo fmt --all

lint:
	cargo clippy

# 集成测试共享 Triton 的统计与 trace 设置，串行执行避免互相影响。
tests:
	cargo nextest run --workspace --no-fail-fast --test-threads=1

update-submodules:
	git submodule update --init --recursive
	git submodule update --remote
	cp deps/triton-core/include/triton/core/tritonbackend.h deps/triton-core/include/triton/core/tritonserver.h triton-ng-sys/include/triton/core/

gen-grpc-client:
	cargo run -p gen-grpc-client

publish:
	cargo publish --workspace $(if $(DRY_RUN),--dry-run)

build:
	cargo build --release
	cargo build --release --package triton-version-backend

download-model:
	mkdir -p models/mnist/1
	mkdir -p models/mnist_onnx/1
	mkdir -p models/identity_gpu/1
	mkdir -p models/decoupled/1
	mkdir -p models/stream_proxy/1
	mkdir -p models/tensor_echo/1
	wget https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx
	mv mnist-12.onnx models/mnist_onnx/1/model.onnx
