# How to run

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package and project manager)

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install dependencies

From the project root:

```bash
uv sync
```

This creates a virtual environment (e.g. `.venv`) and installs dependencies from `pyproject.toml` and `uv.lock`.

## Run Python commands

Scripts in this project import modules from the `src/` directory (`common`, `split_two_machines`). Set `PYTHONPATH` so those imports resolve, then use `uv run` to run Python with the project’s environment:

```bash
export PYTHONPATH=src
```

**Controller** (one terminal):

```bash
uv run control.py --bind 0.0.0.0:9000
```

**Nodes** (one terminal per pipeline part, e.g. 3 parts):

```bash
uv run node.py --index 0 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
uv run node.py --index 1 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
uv run node.py --index 2 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
```

**Client**:

```bash
uv run client.py --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B" --parts 3
```

### One-liner (no persistent export)

You can run a single command with `PYTHONPATH` set only for that run:

```bash
PYTHONPATH=src uv run control.py --bind 0.0.0.0:9000
PYTHONPATH=src uv run node.py --index 0 --parts 3 --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B"
PYTHONPATH=src uv run client.py --control-socket 127.0.0.1:9000 --model "Qwen/Qwen3-0.6B" --parts 3
```

## Other uv commands

- **Run a one-off script with project deps:** `uv run script.py`
- **Add a dependency:** `uv add <package>`
- **Install lockfile only (CI):** `uv sync --frozen`
- **Use a specific Python:** `uv python pin 3.12` then `uv sync`

See [uv documentation](https://docs.astral.sh/uv/) for more.
