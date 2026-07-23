"""gpt2: the running example every topic photographs.

A real, tiny (124M param) causal LM, loaded once per run.py and cached, run
on CPU in eval mode. Determinism: torch.manual_seed(0), model.eval(),
torch.no_grad(), greedy argmax decoding. Oracles assert real token ids,
real logits, real cache tensor shapes -- never sampled prose beyond the
greedy token.

Self-check: run this file directly.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"
SENTENCE = "The cat sat on the mat"  # the fixed running example


def load():
    """Load gpt2 + tokenizer once, deterministic, CPU, eval mode."""
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token  # gpt2 ships no pad token; reuse eos for batching
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model


def forward(model, input_ids, past_key_values=None, attention_mask=None):
    """One real forward pass. Returns (next_token_id, logits[:, -1, :], past)."""
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
    logits = out.logits[:, -1, :]
    next_id = int(torch.argmax(logits, dim=-1).item())
    return next_id, logits, out.past_key_values


if __name__ == "__main__":
    tok, model = load()
    ids = tok(SENTENCE, return_tensors="pt")["input_ids"]
    next_id, logits, past = forward(model, ids)
    print(f"{tok.decode(ids[0])!r} -> {tok.decode([next_id])!r}")
    assert logits.shape[-1] == model.config.vocab_size
    assert past.get_seq_length() == ids.shape[-1]
    assert len(past) == model.config.n_layer
    print("gpt2util self-check: forward pass + cache shape ok.")
