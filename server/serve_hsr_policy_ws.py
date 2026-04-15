#!/usr/bin/env python3
"""
Serve LeRobot pi0 LoRA policy as WebSocket server for HSR client.

Loads everything from a single checkpoint directory:
    checkpoint/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── base_model/   (pi0 base model + tokenizer)
    └── stats.json
"""
import argparse
import logging
import os
from pathlib import Path

from runtime_core.websocket_policy_server import WebsocketPolicyServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve LeRobot pi0 policy for HSR")
    parser.add_argument("--checkpoint-dir", required=True, help="Path to checkpoint directory")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--pytorch-device", default="cuda", help='Torch device (e.g. "cuda", "cpu")')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = str(Path(args.checkpoint_dir).expanduser())
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"checkpoint-dir not found: {checkpoint_dir}")

    from lerobot_pi0_policy import LeRobotPi0Policy

    logging.info("Loading LeRobot pi0 policy from %s ...", checkpoint_dir)
    policy = LeRobotPi0Policy(
        checkpoint_dir=checkpoint_dir,
        device=args.pytorch_device,
    )

    metadata = dict(policy.metadata)
    metadata.update({
        "checkpoint_dir": checkpoint_dir,
        "server_host": args.host,
        "server_port": args.port,
    })

    logging.info("Serving policy on %s:%s", args.host, args.port)
    server = WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=metadata)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
