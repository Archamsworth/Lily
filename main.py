"""Lily – virtual companion.

Entry point: launches the FastAPI server with all subsystems initialised.
"""
import argparse
import sys

import uvicorn

from app.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Lily virtual companion")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument("--host", default=None, help="Override server host")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload (development only)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    server_cfg = cfg["lily"]["server"]

    host = args.host or server_cfg.get("host", "127.0.0.1")
    port = args.port or server_cfg.get("port", 8000)
    reload = args.reload or server_cfg.get("reload", False)

    print(f"Starting Lily on http://{host}:{port}")
    uvicorn.run(
        "app.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    sys.exit(main())
