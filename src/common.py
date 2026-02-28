"""
Shared protocol for nodes, controller, and client.
- Supports TCP (host:port) for network and Unix sockets for local IPC.
- Tensors: 4-byte big-endian length (0 = END), then payload bytes.
- Handshake: 4-byte big-endian length, then UTF-8 JSON (length > 0).
"""

import io
import json
import os
import socket
import struct

# Length 0 means end-of-stream
END_LENGTH = 0
# Default socket directory (Unix IPC); override with DISTRO_SOCK_DIR
SOCK_DIR = os.environ.get("DISTRO_SOCK_DIR", "/tmp")


def parse_address(addr: str) -> tuple[str, str | int, ...]:
    """
    Parse address string. Returns:
    - ("tcp", host, port) for "host:port" or "0.0.0.0:9000"
    - ("unix", path) otherwise (Unix socket path)
    """
    addr = addr.strip()
    if ":" in addr:
        host, _, port_str = addr.rpartition(":")
        if port_str.isdigit():
            return ("tcp", host or "0.0.0.0", int(port_str))
    return ("unix", addr or os.path.join(SOCK_DIR, "control.sock"))


def listen_socket(addr: str) -> socket.socket:
    """Create a listening socket. addr: 'host:port' (TCP) or path (Unix)."""
    kind, *rest = parse_address(addr)
    if kind == "tcp":
        host, port = rest[0], rest[1]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(8)
        return sock
    else:
        path = rest[0]
        if os.path.exists(path):
            os.unlink(path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(path)
        sock.listen(8)
        return sock


def connect_socket(addr: str) -> socket.socket:
    """Connect to addr. addr: 'host:port' (TCP) or path (Unix)."""
    kind, *rest = parse_address(addr)
    if kind == "tcp":
        host, port = rest[0], rest[1]
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        return sock
    else:
        path = rest[0]
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(path)
        return sock


def send_tensor(conn: socket.socket, obj: "torch.Tensor") -> None:
    import torch
    buf = io.BytesIO()
    torch.save(obj.cpu(), buf)
    send_tensor_raw(conn, buf.getvalue())


def recv_tensor_raw(conn: socket.socket) -> bytes | None:
    """Receive length-prefixed payload; returns None on END. No torch dependency."""
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
    return data


def send_tensor_raw(conn: socket.socket, payload: bytes) -> None:
    """Send length-prefixed payload. No torch dependency."""
    conn.sendall(struct.pack("!I", len(payload)))
    conn.sendall(payload)


def recv_tensor(conn: socket.socket) -> "torch.Tensor | None":
    """Receive a tensor. Returns None on END (length 0)."""
    import torch
    data = recv_tensor_raw(conn)
    if data is None:
        return None
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
