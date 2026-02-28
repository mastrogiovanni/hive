"""
Client: tokenizer and orchestration.
- With --control-socket: connect to controller only; controller forwards to the pipeline.
- With --first-socket/--last-socket: connect directly to first and last nodes (legacy).
"""

import argparse
import socket
import sys

import torch

from src.common import connect_socket, recv_tensor, send_end, send_tensor, send_json
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Client: send prompt to pipeline, receive generated tokens.")
    parser.add_argument("--control-socket", type=str, default="", help="Connect to controller (sends tokens, receives responses); if set, first/last sockets ignored")
    parser.add_argument("--first-socket", type=str, default="", help="Socket path of the first pipeline node (omit if using --control-socket)")
    parser.add_argument("--last-socket", type=str, default="", help="Socket path of the last pipeline node (omit if using --control-socket)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B", help="Model id (for tokenizer and pipeline selection)")
    parser.add_argument("--parts", type=int, default=2, help="Pipeline parts (used with --control-socket to select pipeline)")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=20, help="Max new tokens to generate")
    args = parser.parse_args()

    use_control = bool(args.control_socket.strip())
    if not use_control and (not args.first_socket or not args.last_socket):
        print("[client] Either --control-socket or both --first-socket and --last-socket are required", file=sys.stderr)
        sys.exit(1)

    print("[client] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs.input_ids

    if use_control:
        conn = connect_socket(args.control_socket.strip())
        send_json(conn, {"type": "client", "model": args.model, "parts": args.parts})
        print("[client] Connected to controller")
    else:
        sock_first = connect_socket(args.first_socket)
        sock_last = connect_socket(args.last_socket)

    eos_token_id = tokenizer.eos_token_id

    print("[client] Generating...")
    for _ in range(args.max_new_tokens):
        if use_control:
            send_tensor(conn, input_ids)
            next_token = recv_tensor(conn)
        else:
            send_tensor(sock_first, input_ids)
            next_token = recv_tensor(sock_last)
        if next_token is None:
            break
        input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    if use_control:
        send_end(conn)
        conn.close()
    else:
        send_end(sock_first)
        sock_first.close()
        sock_last.close()

    text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
    print("[client] Final response:", text)
    print("[client] Done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
