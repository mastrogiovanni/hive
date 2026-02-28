
# Terminal 1 – controller (listen on all interfaces, port 9000)
uv run control.py --bind 0.0.0.0:9000

# Terminals 2–4 – nodes
uv run node.py --index 0 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
uv run node.py --index 1 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
uv run node.py --index 2 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"

# Terminal 5 – client
uv run client.py --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B" --parts 3
