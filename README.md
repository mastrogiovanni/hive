# LLM split across two machines

## Overview

- **`main.py`** – Single-machine: loads the full model and runs standard `generate()`.
- **`split_two_machines.py`** – Same model executed by **two methods** to emulate two machines:
  1. **Machine 1 (encode)**: `input_ids` → transformer layers 0..K → **hidden_states**
  2. **Machine 2 (decode)**: **hidden_states** → layers K+1..N → norm + lm_head → **logits** → sample next token

Generation is a loop: for each new token, run Method 1 on the current sequence, then Method 2 on the resulting hidden states, then sample and append. This shows that an LLM can be served with the model split across two machines (in a real setup, Machine 1 and Machine 2 would run on different hosts and pass `hidden_states` over the network).

## Run

Install dependencies and run commands with **uv**; see **[RUN.md](RUN.md)** for:

- `uv sync` to install dependencies
- `PYTHONPATH=src uv run ...` for controller, nodes, and client

Model: `Qwen/Qwen3-0.6B` (override via `model_id` in the scripts). CUDA used if available.

---

## Pipeline over IPC (node.py + client.py)

A single **`node.py`** script runs any part of the model pipeline. **`client.py`** drives generation by talking to the first and last nodes.

### node.py

- **`--index`**: This node’s stage index (0 .. parts-1).
- **`--parts`**: Number of pipeline stages (how many pieces the model is split into).
- **`--prev-socket`**: Unix socket path for the **previous** stage (this node receives from here). For index 0, this is where the client connects to send `input_ids`.
- **`--next-socket`**: Unix socket path for the **next** stage (this node sends here). For the last index, this is where the client connects to receive `next_token`.

- **Who creates which socket:**  
  - The **first** node (index 0) only **listens** on `prev_socket`; it **connects** to `next_socket`. So it does **not** create the next socket — the **next** node (index 1) creates it by listening on that path as its **prev_socket**.  
  - Similarly, the **last** node listens on both `prev_socket` (for the previous node) and `next_socket` (for the client).  
  So you must **start from the last node** and go backwards (last → … → first), then start the client.

### client.py

- **`--first-socket`**: Socket of the **first** pipeline node (client sends `input_ids` here).
- **`--last-socket`**: Socket of the **last** pipeline node (client receives `next_token` here).

Optional: `--model`, `--prompt`, `--max-new-tokens`.

### Example: 2-part pipeline (parts=2)

Start the **last** node first, then the first (so each can connect to the next):

```bash
export SOCK=/tmp/distro_sock
# Terminal 1 – last node (index 1): listens on next_socket for client, connects to prev_socket
python node.py --index 1 --parts 2 --prev-socket $SOCK/prev1 --next-socket $SOCK/next1

# Terminal 2 – first node (index 0): listens on prev_socket for client, connects to next_socket (= node1’s prev)
python node.py --index 0 --parts 2 --prev-socket $SOCK/prev0 --next-socket $SOCK/prev1

# Terminal 3 – client: sends to first node’s prev, receives from last node’s next
python client.py --first-socket $SOCK/prev0 --last-socket $SOCK/next1
```

Socket paths are arbitrary; `$SOCK_DIR` (default `/tmp`) can be used.

### How to launch (parts=3)

You need one socket between each pair of stages, plus one for the client to receive from the last node. For 3 parts use four paths: `1.sock`, `2.sock`, `3.sock`, `4.sock`.

- **Node 0** creates only **1.sock** (listens there). It **connects** to 2.sock, so it does not create 2.sock.
- **Node 1** creates **2.sock** (listens there). It connects to 3.sock.
- **Node 2** creates **3.sock** and **4.sock** (listens on both). Client receives on 4.sock.

Start in this order (last node first, so each “next” socket exists when the previous node connects):

```bash
# Terminal 1 – last node (index 2): creates 3.sock and 4.sock, listens on both
uv run node.py --index 2 --parts 3 --prev-socket 3.sock --next-socket 4.sock --model "Qwen/Qwen3-0.6B"

# Terminal 2 – middle node (index 1): creates 2.sock, connects to 3.sock
uv run node.py --index 1 --parts 3 --prev-socket 2.sock --next-socket 3.sock --model "Qwen/Qwen3-0.6B"

# Terminal 3 – first node (index 0): creates 1.sock, connects to 2.sock
uv run node.py --index 0 --parts 3 --prev-socket 1.sock --next-socket 2.sock --model "Qwen/Qwen3-0.6B"

# Terminal 4 – client: connects to 1.sock (first node) and 4.sock (last node)
uv run client.py --first-socket 1.sock --last-socket 4.sock --model "Qwen/Qwen3-0.6B"
```

Summary: **last node first**, then middle, then first, then client. Each node’s `--next-socket` is the **next** node’s `--prev-socket`.

---

## Controller (control.py)

A **controller** exposes a **single socket**. Nodes and clients connect to it; the controller routes tokens through the pipeline in order.

- **Nodes** connect and **register** with `model`, `parts`, and `index`. No prev/next sockets: each node only talks to the controller.
- **Clients** connect and send tokens (input_ids per step); the controller forwards to node 0, then node 1, … then sends the last node’s output (next_token) back to the client.

### control.py

- **`--bind`**: Bind address: **`host:port`** for TCP (default **`0.0.0.0:9000`**), or a path for Unix socket.

### node.py with controller

- **`--control-socket`**: Connect to the controller and register; **`--prev-socket`** and **`--next-socket`** are ignored.

### client.py with controller

- **`--control-socket`**: Connect to the controller; **`--first-socket`** and **`--last-socket`** are ignored.
- **`--model`** and **`--parts`** (default 2) select which registered pipeline to use.

### How to launch with controller (e.g. parts=3)

Use **TCP** so the controller can run on a public server and nodes/clients connect over the network. Start the controller first, then the nodes in any order, then the client.

**Local (TCP):**
```bash
# Terminal 1 – controller (listen on all interfaces, port 9000)
uv run control.py --bind 0.0.0.0:9000

# Terminals 2–4 – nodes
uv run node.py --index 0 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
uv run node.py --index 1 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
uv run node.py --index 2 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"

# Terminal 5 – client
uv run client.py --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B" --parts 3
```

**Unix socket (local only):** use a path for `--bind` and `--control-socket` (e.g. `control.sock`).

The controller only accepts a client when the pipeline for that `(model, parts)` is complete (all indices 0 .. parts−1 registered). Nodes stay alive after a client disconnects, so multiple clients can use the same pipeline.

---

## Docker

Two images: **controller** (minimal, no GPU) and **node** (model + GPU support). The controller listens on TCP so it can run on a public server; nodes and clients connect to `host:port`.

### Build

```bash
cd /path/to/distro

# Controller (small, no torch)
docker build -f docker/Dockerfile.controller -t distro-controller .

# Node (PyTorch + CUDA; use for GPU)
docker build -f docker/Dockerfile.node -t distro-node .
```

Build with Docker BuildKit for cache (e.g. `DOCKER_BUILDKIT=1 docker build ...`).

### Run

**1. Start the controller** (e.g. on a public server; expose port 9000):

```bash
docker run -d --name controller -p 9000:9000 distro-controller
```

**2. Start nodes** (one per pipeline part). Each node must reach the controller at `host:port`. Use `--gpus all` for GPU.

On the same host as the controller:
```bash
docker run -d --name node0 --link controller \
  -e CONTROLLER_ADDRESS=controller:9000 \
  -e NODE_INDEX=0 -e NODE_PARTS=3 \
  -e NODE_MODEL="Qwen/Qwen3-0.6B" \
  --gpus all \
  distro-node

docker run -d --name node1 --link controller \
  -e CONTROLLER_ADDRESS=controller:9000 \
  -e NODE_INDEX=1 -e NODE_PARTS=3 \
  -e NODE_MODEL="Qwen/Qwen3-0.6B" \
  --gpus all \
  distro-node

docker run -d --name node2 --link controller \
  -e CONTROLLER_ADDRESS=controller:9000 \
  -e NODE_INDEX=2 -e NODE_PARTS=3 \
  -e NODE_MODEL="Qwen/Qwen3-0.6B" \
  --gpus all \
  distro-node
```

If nodes run on another machine, set `CONTROLLER_ADDRESS` to the controller’s public host and port (e.g. `controller.example.com:9000`). No `--link` needed.

**3. Run the client** (from your laptop or same host; point to the controller’s address):

```bash
# If controller is on the same host
uv run client.py --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B" --parts 3

# If controller is remote
uv run client.py --control-socket YOUR_CONTROLLER_HOST:9000 --model "Qwen/Qwen3-0.6B" --parts 3
```

### Node image env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTROLLER_ADDRESS` | `controller:9000` | Controller `host:port` (TCP). |
| `NODE_INDEX` | `0` | Pipeline stage index (0 .. parts−1). |
| `NODE_PARTS` | `2` | Total number of pipeline parts. |
| `NODE_MODEL` | `Qwen/Qwen3-0.6B` | Model id. |

Override at runtime: `docker run -e CONTROLLER_ADDRESS=192.168.1.10:9000 -e NODE_INDEX=1 ... distro-node`.
