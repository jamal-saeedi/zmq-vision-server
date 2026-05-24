"""Click CLI: serve and infer commands for zmq_inference."""

from __future__ import annotations

import asyncio
import signal
import sys

import click

from zmq_inference.core.config import ClientConfig, ServerConfig
from zmq_inference.core.logging import setup_logging


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Logging verbosity.",
)
def cli(log_level: str) -> None:
    """ZMQ Inference – serve models and run inference over ZMQ."""
    setup_logging(level=log_level)


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option(
    "--backend",
    required=True,
    type=click.Choice(["torch", "tf"], case_sensitive=False),
    help="Model backend.",
)
@click.option("--host", default=None, help="Bind host (overrides ZMQ_HOST).")
@click.option("--port", default=None, type=int, help="Bind port (overrides ZMQ_PORT).")
@click.option("--model", default=None, help="timm model name (torch backend only).")
@click.option(
    "--device",
    default=None,
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    help="Compute device (torch backend). Overrides ZMQ_DEVICE.",
)
@click.option("--top-k", default=3, show_default=True, help="Number of top predictions.")
def serve_cmd(
    backend: str,
    host: str | None,
    port: int | None,
    model: str | None,
    device: str | None,
    top_k: int,
) -> None:
    """Start an inference server.

    \b
    Examples:
      zmq-inference serve --backend torch --model mobilenetv2_100
      zmq-inference serve --backend tf --port 5577
    """
    config = ServerConfig()
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if model is not None:
        config.model_name = model
    if device is not None:
        config.device = device
    config.top_k = top_k

    if backend.lower() == "torch":
        try:
            from zmq_inference.server.torch_server import TorchInferenceServer
        except ImportError:
            click.echo(
                "PyTorch backend unavailable. Install with: pip install zmq-inference[torch]",
                err=True,
            )
            sys.exit(1)
        server = TorchInferenceServer(config)
    else:
        try:
            from zmq_inference.server.tf_server import TFInferenceServer
        except ImportError:
            click.echo(
                "TensorFlow backend unavailable. Install with: pip install zmq-inference[tf]",
                err=True,
            )
            sys.exit(1)
        server = TFInferenceServer(config)

    click.echo(
        f"Starting {backend.upper()} server on {config.bind_address} "
        f"(model={config.model_name}, device={config.device})"
    )

    server.start()

    # Graceful shutdown on SIGINT/SIGTERM.
    def _shutdown(sig: int, _frame: object) -> None:
        click.echo("\nShutting down…")
        server.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    server.join()


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------


@cli.command("infer")
@click.argument("image_path", type=click.Path(exists=True, readable=True))
@click.option("--host", default=None, help="Server host (overrides ZMQ_SERVER_HOST).")
@click.option("--port", default=None, type=int, help="Server port (overrides ZMQ_PORT).")
@click.option("--top-k", default=3, show_default=True, help="How many predictions to show.")
@click.option(
    "--async",
    "use_async",
    is_flag=True,
    default=False,
    help="Use the async client.",
)
@click.option(
    "--timeout",
    default=None,
    type=int,
    help="Timeout in ms (overrides ZMQ_TIMEOUT_MS).",
)
def infer_cmd(
    image_path: str,
    host: str | None,
    port: int | None,
    top_k: int,
    use_async: bool,
    timeout: int | None,
) -> None:
    """Run inference on IMAGE_PATH and print predictions.

    \b
    Examples:
      zmq-inference infer cat.jpg
      zmq-inference infer cat.jpg --host 192.168.1.10 --top-k 5
      zmq-inference infer cat.jpg --async
    """
    config = ClientConfig()
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    if timeout is not None:
        config.timeout_ms = timeout

    if use_async:
        _run_async(image_path, config, top_k)
    else:
        _run_sync(image_path, config, top_k)


def _run_sync(image_path: str, config: ClientConfig, top_k: int) -> None:
    from zmq_inference.client.sync_client import SyncClient

    with SyncClient(config) as client:
        try:
            response = client.infer(image_path)
        except TimeoutError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)

    _print_response(response, top_k)


def _run_async(image_path: str, config: ClientConfig, top_k: int) -> None:
    from zmq_inference.client.async_client import AsyncClient

    async def _inner() -> None:
        async with AsyncClient(config) as client:
            try:
                response = await client.infer(image_path)
            except TimeoutError as exc:
                click.echo(f"Error: {exc}", err=True)
                sys.exit(1)
        _print_response(response, top_k)

    asyncio.run(_inner())


def _print_response(response: object, top_k: int) -> None:
    from zmq_inference.core.protocol import InferenceResponse

    assert isinstance(response, InferenceResponse)
    if not response.ok:
        click.echo(f"Server error: {response.error}", err=True)
        sys.exit(1)

    click.echo(f"Request ID : {response.request_id}")
    click.echo(f"Elapsed    : {response.elapsed_ms:.1f} ms")
    click.echo("Predictions:")
    for i, pred in enumerate(response.predictions[:top_k], start=1):
        click.echo(f"  {i}. {pred.label:40s} {pred.confidence:.5f}")


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@cli.command("health")
@click.option("--host", default=None, help="Server host.")
@click.option("--port", default=None, type=int, help="Server port.")
@click.option("--timeout", default=5000, show_default=True, help="Timeout in ms.")
def health_cmd(host: str | None, port: int | None, timeout: int) -> None:
    """Check whether the inference server is reachable.

    Sends a minimal 1x1 pixel image and reports success or timeout.

    \b
    Examples:
      zmq-inference health
      zmq-inference health --host 10.0.0.5 --port 5576
    """
    import io

    from PIL import Image

    from zmq_inference.client.sync_client import SyncClient
    from zmq_inference.utils.image import encode_image

    config = ClientConfig()
    if host is not None:
        config.host = host
    if port is not None:
        config.port = port
    config.timeout_ms = timeout
    config.max_retries = 1

    # Build a tiny 1x1 white RGB image as the health-check payload.
    tiny = Image.new("RGB", (1, 1), color=(255, 255, 255))

    click.echo(f"Checking {config.server_address} …")
    with SyncClient(config) as client:
        try:
            response = client.infer(tiny)
            click.echo(
                click.style(
                    f"OK  – server responded in {response.elapsed_ms:.1f} ms",
                    fg="green",
                )
            )
        except TimeoutError:
            click.echo(
                click.style(
                    f"TIMEOUT – no response within {timeout} ms",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)


if __name__ == "__main__":
    cli()
