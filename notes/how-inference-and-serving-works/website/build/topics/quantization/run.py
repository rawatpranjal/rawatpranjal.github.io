"""Quantization: real on-disk footprint, measured. gpt2's actual state_dict
is saved to a temp file in fp32, then again after model.half() in fp16, and
the oracle asserts on the REAL byte sizes read back off disk -- never on the
arithmetic identity fp16_bytes == fp32_bytes // 2, which would be true by
construction and prove nothing about the real model.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
root = HERE
while not (root / "lib").is_dir():
    root = root.parent
sys.path.insert(0, str(root / "lib"))

from gpt2util import load  # noqa: E402
from langviz import clear, draw_card, draw_scorecard  # noqa: E402

FIGURES = HERE / "figures"
TITLE = "Quantization"
CONTEXT = "The cat sat on the"


def mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def main():
    clear(FIGURES)
    tok, model = load()

    n_params = sum(p.numel() for p in model.parameters())

    # forward-pass timing, on the real fp32 model, before any dtype change
    ctx_ids = tok(CONTEXT, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        model(input_ids=ctx_ids)  # warm up
        t0 = time.perf_counter()
        for _ in range(5):
            model(input_ids=ctx_ids)
        elapsed_ms = (time.perf_counter() - t0) / 5 * 1000

    # the real measurement: save the actual state_dict to disk in fp32, then
    # again after model.half() in fp16, and read back the real byte counts.
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "gpt2_fp32.pt"
        torch.save(model.state_dict(), fp32_path)
        fp32_bytes = fp32_path.stat().st_size

        model.half()
        fp16_path = Path(tmp) / "gpt2_fp16.pt"
        torch.save(model.state_dict(), fp16_path)
        fp16_bytes = fp16_path.stat().st_size
        # both files live only inside this TemporaryDirectory; removed on exit

    ratio = fp16_bytes / fp32_bytes

    draw_card(
        f"n_params = sum(p.numel() for p in model.parameters())\n"
        f"n_params = {n_params:,}\n\n"
        "the same count either way -- quantization changes how each\n"
        "number is STORED, never how many numbers there are.",
        FIGURES / "step-01.png",
        TITLE,
        subtitle="the real gpt2 parameter count",
        note="model.parameters() walks the real loaded nn.Module, not a spec sheet.",
    )

    draw_scorecard(
        [
            {
                "label": "fp32 state_dict on disk",
                "cells": [f"{n_params:,}", f"{mb(fp32_bytes):,.1f} MB"],
                "verdict": "pass",
            },
            {
                "label": "fp16 state_dict on disk",
                "cells": [f"{n_params:,}", f"{mb(fp16_bytes):,.1f} MB"],
                "verdict": "pass",
            },
        ],
        FIGURES / "step-02.png",
        TITLE,
        columns=["params", "measured footprint"],
        note=f"torch.save(model.state_dict(), path), read back with os.stat: fp16/fp32 = {ratio:.4f}.",
    )

    draw_card(
        f"fp32 forward pass, CPU, 5-call average: {elapsed_ms:.1f} ms\n\n"
        "shown for context only -- machine load varies run to run,\n"
        "so the oracle below asserts the measured file sizes, never this number.",
        FIGURES / "step-03.png",
        TITLE,
        subtitle="a timing, not a claim",
        note="fp16 halves memory traffic; whether that halves wall-clock depends on the hardware, not asserted here.",
    )

    # ---- oracle: real measured on-disk bytes, not arithmetic identity ----
    assert n_params > 0
    assert fp32_bytes > 0 and fp16_bytes > 0
    assert fp32_bytes > fp16_bytes, (
        "fp32 state_dict must measure larger on disk than fp16"
    )
    assert 0.48 < ratio < 0.52, (
        f"fp16 state_dict should measure close to half the fp32 state_dict "
        f"on disk, got ratio {ratio:.4f} (fp32={fp32_bytes} fp16={fp16_bytes})"
    )

    figs = sorted(FIGURES.glob("*.png"))
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    print(
        f"{len(figs)} figures, {n_params:,} real params, "
        f"fp32={mb(fp32_bytes):,.1f}MB fp16={mb(fp16_bytes):,.1f}MB measured on disk "
        f"(ratio {ratio:.4f}), timing {elapsed_ms:.1f}ms shown but not asserted. All checks passed."
    )


if __name__ == "__main__":
    main()
