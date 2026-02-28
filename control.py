"""
Controller: single socket for nodes and clients.
- Nodes connect and register (model, parts, index).
- Clients connect and send tokens / receive responses; controller forwards through the pipeline in order.
"""

import argparse
import json
import os
import socket
import sys
import threading

from common import recv_json, recv_tensor, send_end, send_tensor


def pipeline_complete(pipeline: dict, parts: int) -> bool:
    if len(pipeline) != parts:
        return False
    for i in range(parts):
        if i not in pipeline:
            return False
    return True


def client_handler(client_conn: socket.socket, pipeline_key: tuple, pipelines: dict) -> None:
    """Run generation loop: recv input_ids from client, forward through pipeline, send next_token to client."""
    model, parts = pipeline_key
    nodes = pipelines[pipeline_key]
    try:
        while True:
            input_ids = recv_tensor(client_conn)
            if input_ids is None:
                # Client sent END; do not send END to nodes so they stay alive for the next client
                break
            data = input_ids
            for i in range(parts):
                send_tensor(nodes[i], data)
                data = recv_tensor(nodes[i])
                if data is None:
                    break
            if data is None:
                break
            send_tensor(client_conn, data)
    except (BrokenPipeError, ConnectionResetError, OSError):
        pass
    finally:
        try:
            client_conn.close()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Controller: nodes and clients connect to one socket.")
    parser.add_argument("--socket", type=str, default="control.sock", help="Unix socket path")
    args = parser.parse_args()

    sock_path = args.socket
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(8)
    print(f"[control] Listening on {sock_path}")

    # (model, parts) -> { index: conn }
    pipelines: dict[tuple[str, int], dict[int, socket.socket]] = {}
    pipelines_lock = threading.Lock()

    while True:
        conn, _ = server.accept()
        try:
            msg = recv_json(conn)
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
            t = threading.Thread(target=client_handler, args=(conn, key, pipelines))
            t.daemon = True
            t.start()
        else:
            conn.close()


if __name__ == "__main__":
    main()
