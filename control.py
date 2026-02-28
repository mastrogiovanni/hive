"""
Controller: single socket for nodes and clients.
- Nodes connect and register (model, parts, index).
- Clients connect and send tokens / receive responses; controller forwards through the pipeline in order.
"""

import argparse
import json
import socket
import sys
import threading

from common import listen_socket, recv_json, recv_tensor_raw, send_tensor_raw


def pipeline_complete(pipeline: dict, parts: int) -> bool:
    if len(pipeline) != parts:
        return False
    for i in range(parts):
        if i not in pipeline:
            return False
    return True


def client_handler(client_conn: socket.socket, pipeline_key: tuple, pipelines: dict) -> None:
    """Run generation loop: recv tensor bytes from client, forward through pipeline, send result to client."""
    model, parts = pipeline_key
    nodes = pipelines[pipeline_key]
    try:
        while True:
            data = recv_tensor_raw(client_conn)
            if data is None:
                break
            for i in range(parts):
                send_tensor_raw(nodes[i], data)
                data = recv_tensor_raw(nodes[i])
                if data is None:
                    break
            if data is None:
                break
            send_tensor_raw(client_conn, data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            client_conn.close()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Controller: nodes and clients connect here.")
    parser.add_argument(
        "--bind",
        type=str,
        default="0.0.0.0:9000",
        help="Bind address: 'host:port' for TCP (e.g. 0.0.0.0:9000), or path for Unix socket",
    )
    args = parser.parse_args()

    server = listen_socket(args.bind)
    print(f"[control] Listening on {args.bind}")

    # (model, parts) -> { index: conn }
    pipelines: dict[tuple[str, int], dict[int, socket.socket]] = {}
    pipelines_lock = threading.Lock()

    while True:
        conn, _ = server.accept()
        try:
            msg = recv_json(conn)
            print(f"[control] Received message: {msg}")
        except (json.JSONDecodeError, OSError):
            conn.close()
            continue
        if not msg or "type" not in msg:
            conn.close()
            continue

        if msg["type"] == "node":
            model = msg.get("model", "Qwen/Qwen3-0.6B")
            parts = int(msg.get("parts", 2))
            index = int(msg.get("index", 0))
            if index < 0 or index >= parts:
                conn.close()
                continue
            with pipelines_lock:
                key = (model, parts)
                if key not in pipelines:
                    pipelines[key] = {}
                pipelines[key][index] = conn
            print(f"[control] Registered node model={model!r} parts={parts} index={index}")

        elif msg["type"] == "client":
            model = msg.get("model", "Qwen/Qwen3-0.6B")
            parts = int(msg.get("parts", 2))
            key = (model, parts)
            with pipelines_lock:
                if not pipeline_complete(pipelines.get(key, {}), parts):
                    print(f"[control] Client rejected: pipeline {key} not complete")
                    conn.close()
                    continue
                nodes = pipelines[key]
            print(f"[control] Client connected model={model!r} parts={parts}")
            t = threading.Thread(target=client_handler, args=(conn, key, pipelines))
            t.daemon = True
            t.start()
        else:
            conn.close()


if __name__ == "__main__":
    main()
