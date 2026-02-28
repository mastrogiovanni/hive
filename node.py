"""
Generic pipeline node: runs one part of the model (index of parts).
- index 0: LISTENS on prev_socket only (creates that file). CONNECTS to next_socket (does not create it; next node creates it as its prev_socket).
- index parts-1: CONNECTS to prev_socket, LISTENS on next_socket (creates both).
- middle: CONNECTS to both (creates neither; prev node creates next_socket as its listening address).
Start order: last node first, then backwards to index 0, then client.
"""

import argparse
import os
import socket
import sys

import torch

from common import connect_socket, recv_json, recv_tensor, send_end, send_json, send_tensor
from split_two_machines import get_model_part
from transformers import AutoModelForCausalLM


def run_controller_mode(conn: socket.socket, part, index: int, parts: int, device: str) -> None:
    """Loop: recv tensor from controller, compute, send result. Exit on END."""
    with torch.no_grad():
        while True:
            data = recv_tensor(conn)
            if data is None:
                break
            data = data.to(device)
            out = part(data)
            if index == parts - 1:
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                send_tensor(conn, next_token)
            else:
                send_tensor(conn, out)
    conn.close()


def run_peer_mode(part, index: int, parts: int, device: str, prev_socket_path: str, next_socket_path: str) -> None:
    """Original mode: connect/listen to prev and next sockets."""
    is_first = index == 0
    is_last = index == parts - 1

    if is_first:
        if os.path.exists(prev_socket_path):
            os.unlink(prev_socket_path)
        server_prev = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_prev.bind(prev_socket_path)
        server_prev.listen(1)
        print(f"[node{index}] Listening on prev_socket={prev_socket_path}")
        conn_prev, _ = server_prev.accept()
        print(f"[node{index}] Client connected (prev)")
    else:
        conn_prev = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn_prev.connect(prev_socket_path)
        print(f"[node{index}] Connected to previous node at {prev_socket_path}")

    if is_last:
        if os.path.exists(next_socket_path):
            os.unlink(next_socket_path)
        server_next = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_next.bind(next_socket_path)
        server_next.listen(1)
        print(f"[node{index}] Listening on next_socket={next_socket_path} (client will receive)")
        conn_next, _ = server_next.accept()
        print(f"[node{index}] Client connected (next)")
    else:
        conn_next = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn_next.connect(next_socket_path)
        print(f"[node{index}] Connected to next node at {next_socket_path}")

    with torch.no_grad():
        while True:
            data = recv_tensor(conn_prev)
            if data is None:
                if not is_last:
                    send_end(conn_next)
                break
            data = data.to(device)
            out = part(data)
            if is_last:
                next_token = out[:, -1, :].argmax(dim=-1, keepdim=True)
                send_tensor(conn_next, next_token)
            else:
                send_tensor(conn_next, out)

    conn_prev.close()
    conn_next.close()
    if is_first:
        server_prev.close()
    if is_last:
        server_next.close()


def main():
    parser = argparse.ArgumentParser(description="Pipeline node: run one part of the model.")
    parser.add_argument("--index", type=int, required=True, help="This node's index in the pipeline (0 .. parts-1)")
    parser.add_argument("--parts", type=int, required=True, help="Total number of pipeline parts")
    parser.add_argument("--prev-socket", type=str, default="", help="Socket path for previous node (omit if using --control-socket)")
    parser.add_argument("--next-socket", type=str, default="", help="Socket path for next node (omit if using --control-socket)")
    parser.add_argument("--control-socket", type=str, default="", help="If set, connect to controller and register (model, parts, index); prev/next sockets ignored")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model id")
    args = parser.parse_args()

    index = args.index
    parts = args.parts
    control_socket_path = args.control_socket.strip()
    prev_socket_path = args.prev_socket
    next_socket_path = args.next_socket

    if index < 0 or index >= parts:
        print(f"[node{index}] Invalid index {index} for parts={parts}", file=sys.stderr)
        sys.exit(1)

    if control_socket_path and (not prev_socket_path or not next_socket_path):
        pass  # controller mode: prev/next not needed
    elif not control_socket_path and (not prev_socket_path or not next_socket_path):
        print("[node] Either --control-socket or both --prev-socket and --next-socket are required", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[node{index}] Loading model and building part {index}/{parts}...")
    full_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map=device,
    )
    part = get_model_part(full_model, parts, index).to(device).to(dtype).eval()
    del full_model
    if device == "cuda":
        torch.cuda.empty_cache()

    if control_socket_path:
        conn = connect_socket(control_socket_path)
        send_json(conn, {"type": "node", "model": args.model, "parts": parts, "index": index})
        print(f"[node{index}] Registered with controller at {control_socket_path}")
        run_controller_mode(conn, part, index, parts, device)
    else:
        run_peer_mode(part, index, parts, device, prev_socket_path, next_socket_path)

    print(f"[node{index}] Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
