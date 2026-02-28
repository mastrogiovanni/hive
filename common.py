"""
Shared IPC protocol for nodes, controller, and client.
- Tensors: 4-byte big-endian length (0 = END), then payload bytes.
- Handshake: 4-byte big-endian length, then UTF-8 JSON (length > 0).
"""

import io
import json
import os
import socket
import struct

import torch

# Length 0 means end-of-stream
END_LENGTH = 0
# Default socket directory (Unix IPC); override with DISTRO_SOCK_DIR
SOCK_DIR = os.environ.get("DISTRO_SOCK_DIR", "/tmp")


def send_tensor(conn: socket.socket, obj: torch.Tensor) -> None:
    buf = io.BytesIO()
    torch.save(obj.cpu(), buf)
    payload = buf.getvalue()
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def recv_tensor(conn: socket.socket) -> torch.Tensor | None:
    """Receive a tensor. Returns None on END (length 0)."""
    raw = conn.recv(4)
    if not raw or len(raw) < 4:
        return None
    (length,) = struct.unpack("!I", raw)
    if length == END_LENGTH:
        return None
    data = b""
    while len(data) < length:
        chunk = conn.recv(min(1 << 20, length - len(data)))
        if not chunk:
            return None
        data += chunk
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def send_end(conn: socket.socket) -> None:
    conn.sendall(struct.pack("!I", END_LENGTH))


def send_json(conn: socket.socket, obj: dict) -> None:
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def recv_json(conn: socket.socket) -> dict | None:
    raw = conn.recv(4)
    if not raw or len(raw) < 4:
        return None
    (length,) = struct.unpack("!I", raw)
    if length == 0:
        return None
    data = b""
    while len(data) < length:
        chunk = conn.recv(min(1 << 20, length - len(data)))
        if not chunk:
            return None
        data += chunk
    return json.loads(data.decode("utf-8"))


def recv_exact(conn: socket.socket, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            break
        data += chunk
    return data
