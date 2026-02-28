"""
Emulates running an LLM split across two machines.

Instead of a single generate(), execution uses two methods:
  - Machine 1 (encode): input_ids -> hidden_states
  - Machine 2 (decode): hidden_states -> logits, then sample next token

This script runs both stages in one process to prove that splitting the model
produces valid generation. For real deployment, Machine 1 and Machine 2
would run on different hosts and pass hidden_states over the network.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask


def get_decoder_layers(model):
    """Get the list of transformer layers from a causal LM (works for Qwen, LLaMA, etc.)."""
    base = model.model if hasattr(model, "model") else model
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "h"):  # GPT-2 style
        return base.h
    raise AttributeError("Could not find decoder layers on model")


def get_embed_norm_lmhead(model):
    """Get embed, final norm and lm_head from the model."""
    base = model.model if hasattr(model, "model") else model
    embed = getattr(base, "embed_tokens", None) or getattr(base, "wte", None)
    norm = getattr(base, "norm", None)
    lm_head = model.lm_head if hasattr(model, "lm_head") else getattr(base, "lm_head", None)
    return embed, norm, lm_head


def get_rotary_emb(model):
    """Get rotary embedding module (for Qwen3, LLaMA, etc.) if present."""
    base = model.model if hasattr(model, "model") else model
    return getattr(base, "rotary_emb", None)


def _prepare_position_embeddings_and_mask(rotary_emb, config, hidden, device, has_sliding_layers=False):
    """Build position_ids, position_embeddings (cos, sin), and causal mask for Qwen3-style models."""
    seq_len = hidden.shape[1]
    batch_size = hidden.shape[0]
    cache_position = torch.arange(seq_len, device=device, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
    position_embeddings = rotary_emb(hidden, position_ids) if rotary_emb is not None else None

    mask_kwargs = {
        "config": config,
        "input_embeds": hidden,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
    if has_sliding_layers:
        from transformers.masking_utils import create_sliding_window_causal_mask
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    return position_ids, position_embeddings, causal_mask_mapping


# class ModelPart1(nn.Module):
#     """
#     First half of the model (Machine 1).
#     Input: input_ids -> Output: hidden_states (to be sent to Machine 2).
#     """

#     def __init__(self, model, split_layer_idx):
#         super().__init__()
#         base = model.model if hasattr(model, "model") else model
#         self.embed, _, _ = get_embed_norm_lmhead(model)
#         self.layers = get_decoder_layers(model)[:split_layer_idx]
#         self.rotary_emb = get_rotary_emb(model)
#         self.config = model.config
#         self.has_sliding_layers = getattr(self.config, "layer_types", None) and "sliding_attention" in self.config.layer_types

#     def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
#         hidden = self.embed(input_ids)
#         device = hidden.device
#         position_ids, position_embeddings, causal_mask_mapping = _prepare_position_embeddings_and_mask(
#             self.rotary_emb, self.config, hidden, device, self.has_sliding_layers
#         )
#         for i, layer in enumerate(self.layers):
#             layer_attn_type = getattr(layer, "attention_type", "full_attention")
#             layer_mask = causal_mask_mapping.get(layer_attn_type, causal_mask_mapping["full_attention"])
#             out = layer(
#                 hidden,
#                 attention_mask=layer_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 use_cache=use_cache,
#                 position_embeddings=position_embeddings,
#                 **kwargs,
#             )
#             hidden = out[0] if isinstance(out, (list, tuple)) else out
#         return hidden


# class ModelPart2(nn.Module):
#     """
#     Second half of the model (Machine 2).
#     Input: hidden_states (from Machine 1) -> Output: logits.
#     """

#     def __init__(self, model, split_layer_idx):
#         super().__init__()
#         self.layers = get_decoder_layers(model)[split_layer_idx:]
#         _, self.norm, self.lm_head = get_embed_norm_lmhead(model)
#         self.rotary_emb = get_rotary_emb(model)
#         self.config = model.config
#         self.has_sliding_layers = getattr(self.config, "layer_types", None) and "sliding_attention" in self.config.layer_types

#     def forward(self, hidden_states, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
#         device = hidden_states.device
#         position_ids, position_embeddings, causal_mask_mapping = _prepare_position_embeddings_and_mask(
#             self.rotary_emb, self.config, hidden_states, device, self.has_sliding_layers
#         )
#         for i, layer in enumerate(self.layers):
#             layer_attn_type = getattr(layer, "attention_type", "full_attention")
#             layer_mask = causal_mask_mapping.get(layer_attn_type, causal_mask_mapping["full_attention"])
#             out = layer(
#                 hidden_states,
#                 attention_mask=layer_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 use_cache=use_cache,
#                 position_embeddings=position_embeddings,
#                 **kwargs,
#             )
#             hidden_states = out[0] if isinstance(out, (list, tuple)) else out
#         if self.norm is not None:
#             hidden_states = self.norm(hidden_states)
#         logits = self.lm_head(hidden_states)
#         return logits


class ModelPartGeneric(nn.Module):
    """
    Generic pipeline part: one of N pieces of the model.
    - index 0: embed + layers[0:k] (input_ids -> hidden_states)
    - index i (0 < i < parts-1): layers[k:m] (hidden_states -> hidden_states)
    - index parts-1: layers[m:] + norm + lm_head (hidden_states -> logits)
    """

    def __init__(
        self,
        model,
        start_layer: int,
        end_layer: int,
        *,
        include_embed: bool = False,
        include_norm_lm_head: bool = False,
    ):
        super().__init__()
        layers = get_decoder_layers(model)
        self.embed = None
        if include_embed:
            self.embed, _, _ = get_embed_norm_lmhead(model)
        self.layers = layers[start_layer:end_layer]
        self.norm = None
        self.lm_head = None
        if include_norm_lm_head:
            _, self.norm, self.lm_head = get_embed_norm_lmhead(model)
        self.rotary_emb = get_rotary_emb(model)
        self.config = model.config
        self.has_sliding_layers = getattr(self.config, "layer_types", None) and "sliding_attention" in self.config.layer_types
        self._include_embed = include_embed
        self._include_norm_lm_head = include_norm_lm_head

    def forward(self, x, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        if self._include_embed:
            hidden = self.embed(x)
        else:
            hidden = x
        device = hidden.device
        position_ids, position_embeddings, causal_mask_mapping = _prepare_position_embeddings_and_mask(
            self.rotary_emb, self.config, hidden, device, self.has_sliding_layers
        )
        for layer in self.layers:
            layer_attn_type = getattr(layer, "attention_type", "full_attention")
            layer_mask = causal_mask_mapping.get(layer_attn_type, causal_mask_mapping["full_attention"])
            out = layer(
                hidden,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden = out[0] if isinstance(out, (list, tuple)) else out
        if self._include_norm_lm_head:
            if self.norm is not None:
                hidden = self.norm(hidden)
            return self.lm_head(hidden)
        return hidden


def get_model_part(model, parts: int, index: int) -> nn.Module:
    """Return the pipeline part for this index (0 .. parts-1)."""
    layers = get_decoder_layers(model)
    n = len(layers)
    if index < 0 or index >= parts or parts < 1:
        raise ValueError("index must be in [0, parts-1] and parts >= 1")
    start = n * index // parts
    end = n * (index + 1) // parts
    include_embed = index == 0
    include_norm_lm_head = index == parts - 1
    return ModelPartGeneric(
        model,
        start,
        end,
        include_embed=include_embed,
        include_norm_lm_head=include_norm_lm_head,
    )


# def generate_split(
#     part1: ModelPart1,
#     part2: ModelPart2,
#     input_ids: torch.Tensor,
#     max_new_tokens: int = 20,
#     eos_token_id: int = None,
#     pad_token_id: int = None,
# ):
#     """
#     Generate tokens using the two-part model (emulates two machines).
#     Each step: Machine 1 encodes current sequence -> Machine 2 decodes to logits -> sample next token.
#     """
#     part1.eval()
#     part2.eval()
#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             # --- Machine 1: encode current sequence -> hidden_states ---
#             hidden = part1(input_ids)

#             # --- Machine 2: decode last position -> logits ---
#             logits = part2(hidden)

#             # Next token from last position
#             next_token_logits = logits[:, -1, :]
#             next_token = next_token_logits.argmax(dim=-1, keepdim=True)

#             input_ids = torch.cat([input_ids, next_token], dim=-1)

#             if eos_token_id is not None and (next_token == eos_token_id).all():
#                 break
#         return input_ids


# def main():
#     model_id = "Qwen/Qwen3-0.6B"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float16 if device == "cuda" else torch.float32

#     print("Loading full model (used only to build Part1 and Part2)...")
#     full_model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         dtype=dtype,
#         device_map=device,
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     layers = get_decoder_layers(full_model)
#     split_idx = len(layers) // 2
#     print(f"Split: first {split_idx} layers = Machine 1, remaining {len(layers) - split_idx} = Machine 2")

#     part1 = ModelPart1(full_model, split_idx).to(device).to(dtype).eval()
#     part2 = ModelPart2(full_model, split_idx).to(device).to(dtype).eval()

#     prompt = "The capital of France is"
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     input_ids = inputs.input_ids

#     print("\n--- Generation via TWO methods (split across two machines) ---")
#     generated = generate_split(
#         part1,
#         part2,
#         input_ids.clone(),
#         max_new_tokens=20,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#     )
#     text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
#     print(text)

#     print("\nDone. The model was executed by two methods (encode on Machine 1, decode on Machine 2).")
#     print("This proves an LLM can be served with the model split across two machines.")


# if __name__ == "__main__":
#     main()
