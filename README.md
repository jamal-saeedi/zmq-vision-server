# zmq-inference

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**zmq-inference** is a high-performance ZMQ ROUTER/DEALER inference server and client library supporting both PyTorch (via [timm](https://github.com/huggingface/pytorch-image-models)) and TensorFlow/Keras backends for ImageNet image classification.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT SIDE                            │
│                                                                 │
│   SyncClient / AsyncClient                                      │
│   (ZMQ DEALER socket, unique identity per client)               │
│         │  send_json(InferenceRequest)                          │
│         │  recv_string(InferenceResponse)                       │
└─────────┼───────────────────────────────────────────────────────┘
          │  TCP  tcp://host:5576
┌─────────┼───────────────────────────────────────────────────────┐
│         ▼           SERVER SIDE                                 │
│   ROUTER frontend (bind tcp://*:5576)                           │
│         │  recv identity + payload                              │
│         ▼                                                        │
│   RequestHandler (Thread per request)                           │
│         │  decode → preprocess → predict                        │
│         ▼                                                        │
│   DEALER backend (inproc://backend)                             │
│         │  send identity + InferenceResponse JSON               │
│         ▼                                                        │
│   ROUTER frontend → TCP → Client                                │
│                                                                 │
│   Backends:  TorchInferenceServer  |  TFInferenceServer         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/jamal-saeedi/zmq-inference-server-client-pytorch-tf.git
cd zmq-inference-server-client-pytorch-tf

# 2. Install with PyTorch backend
pip install -e ".[torch,dev]"

# 3. Start the server
zmq-inference serve --backend torch --model mobilenetv2_100

# 4. Run inference (in another terminal)
zmq-inference infer sample.png
```

---

## Installation

### PyTorch backend (recommended)

```bash
pip install zmq-inference[torch]
```

### TensorFlow backend

```bash
pip install zmq-inference[tf]
```

### Development (tests, linting, type checking)

```bash
pip install zmq-inference[torch,dev]
# or
make install-torch
```

---

## CLI Usage

### Start the server

```bash
# PyTorch with MobileNetV2
zmq-inference serve --backend torch --model mobilenetv2_100

# TensorFlow on port 5577
zmq-inference serve --backend tf --port 5577

# PyTorch, force CPU
zmq-inference serve --backend torch --device cpu

# All options
zmq-inference serve --backend torch \
    --host "*" \
    --port 5576 \
    --model efficientnet_b0 \
    --device auto \
    --top-k 5
```

### Run inference

```bash
# Synchronous (default)
zmq-inference infer path/to/image.jpg

# Async client
zmq-inference infer path/to/image.jpg --async

# Remote server, show top-5
zmq-inference infer image.jpg --host 10.0.0.5 --port 5576 --top-k 5
```

Example output:

```
Request ID : 3f9a1b2c-0d4e-4f5a-8b6c-7d8e9f0a1b2c
Elapsed    : 23.4 ms
Predictions:
  1. tabby                                    0.68432
  2. tiger cat                                0.19217
  3. Egyptian cat                             0.07341
```

### Health check

```bash
zmq-inference health
zmq-inference health --host 10.0.0.5 --timeout 3000
```

---

## Configuration

All settings can be supplied via environment variables (or a `.env` file).

| Variable | Default | Description |
|---|---|---|
| `ZMQ_HOST` | `*` | Server bind host |
| `ZMQ_PORT` | `5576` | TCP port |
| `ZMQ_MODEL` | `mobilenetv2_100` | timm model name (torch backend) |
| `ZMQ_DEVICE` | `auto` | Compute device: `auto`, `cpu`, `cuda` |
| `ZMQ_SERVER_HOST` | `localhost` | Client – server hostname |
| `ZMQ_TIMEOUT_MS` | `30000` | Client receive timeout (ms) |

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

---

## Python API Reference

### Server

```python
from zmq_inference import ServerConfig
from zmq_inference.server.torch_server import TorchInferenceServer

config = ServerConfig(port=5576, model_name="mobilenetv2_100", top_k=3)
server = TorchInferenceServer(config)
server.start()
server.join()  # block until stopped
```

### Synchronous client

```python
from zmq_inference import ClientConfig
from zmq_inference.client.sync_client import SyncClient

config = ClientConfig(host="localhost", port=5576, timeout_ms=10000)

with SyncClient(config) as client:
    response = client.infer("cat.jpg")
    for pred in response.predictions:
        print(f"{pred.label}: {pred.confidence:.4f}")
```

### Async client

```python
import asyncio
from zmq_inference.client.async_client import AsyncClient

async def main():
    async with AsyncClient() as client:
        response = await client.infer("cat.jpg")
        print(response)

asyncio.run(main())
```

### Protocol dataclasses

```python
from zmq_inference.core.protocol import InferenceRequest, InferenceResponse, Prediction

req = InferenceRequest(payload=base64_str)
print(req.to_json())

resp = InferenceResponse.from_json(raw_json)
print(resp.predictions[0].label, resp.elapsed_ms)
```

### Image utilities

```python
from zmq_inference.utils.image import encode_image, decode_image

b64 = encode_image("photo.jpg")        # from file path
b64 = encode_image(pil_image)          # from PIL.Image
img = decode_image(b64)                # returns PIL.Image (RGB)
```

---

## Development

```bash
# Install with all extras
make install-torch

# Run tests with coverage
make test

# Lint
make lint

# Auto-format
make format

# Type check
make type-check

# Clean build artifacts
make clean
```

### Project layout

```
zmq_inference/
├── __init__.py
├── server/
│   ├── base.py          # Abstract BaseInferenceServer (ZMQ ROUTER/DEALER loop)
│   ├── torch_server.py  # PyTorch/timm implementation
│   └── tf_server.py     # TensorFlow/Keras implementation
├── client/
│   ├── base.py          # Abstract BaseInferenceClient
│   ├── sync_client.py   # Synchronous polling client
│   └── async_client.py  # Asyncio client
├── core/
│   ├── config.py        # ServerConfig, ClientConfig dataclasses
│   ├── protocol.py      # InferenceRequest, InferenceResponse, Prediction
│   └── logging.py       # Structured logging setup
├── utils/
│   ├── image.py         # encode_image, decode_image, preprocess_for_display
│   └── labels.py        # load_imagenet_labels, get_label
├── cli/
│   └── main.py          # Click CLI: serve, infer, health
└── data/
    └── imagenet_classes.json
tests/
├── conftest.py
├── test_protocol.py
├── test_image_utils.py
├── test_sync_client.py
└── test_server.py
```

---

## Legacy Scripts

The original flat scripts are preserved for backward compatibility:

| Script | Description |
|---|---|
| `inference_server_torch.py` | Original PyTorch server |
| `inference_server_tf.py` | Original TensorFlow server |
| `client_async.py` | Original asyncio client |
| `client_loop.py` | Original sync polling client |

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`.
3. Make your changes and add tests.
4. Ensure `make test lint format-check type-check` all pass.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License.

