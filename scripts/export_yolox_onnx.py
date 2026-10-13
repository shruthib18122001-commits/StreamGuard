#!/usr/bin/env python3
"""Export a YOLOX checkpoint to a dynamic-batch, NCHW, UNDECODED ONNX file.

Mirrors YOLOX's own tools/export_onnx.py (same model construction, SiLU
replacement, opset 11, tensor names "images"/"output") with two differences:
it calls the public torch.onnx.export instead of the private
torch.onnx._export that newer torch releases may have removed, and it pins the
two settings this repo depends on:

  * dynamic batch axis  -- the dynamic batcher feeds batches of 1..16; a
    static batch-1 export cannot back a min!=max TensorRT profile.
  * decode_in_inference=False -- output is (N, anchors, 85) with raw box
    offsets and sigmoid-activated obj/cls (yolox/models/yolo_head.py, the
    non-training branch concatenates reg_output, obj_output.sigmoid(),
    cls_output.sigmoid()). streamguard/inference/yolox_decode.py applies grid
    and stride itself, so a pre-decoded export would be decoded twice.

Needs a YOLOX source checkout on disk (not installed, not a dependency of this
repo). Usage:
    python scripts/export_yolox_onnx.py --yolox-dir /content/YOLOX --name yolox-tiny \
        --ckpt yolox_tiny.pth --output yolox_tiny.onnx
"""
from __future__ import annotations

import argparse
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yolox-dir", required=True, help="path to a YOLOX git checkout")
    ap.add_argument("--name", required=True, help="e.g. yolox-nano, yolox-tiny, yolox-s, yolox-m")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=11)
    args = ap.parse_args()

    sys.path.insert(0, args.yolox_dir)
    import torch
    from torch import nn
    from yolox.exp import get_exp
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    exp = get_exp(None, args.name)
    model = exp.get_model()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.eval()
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    h, w = exp.test_size
    dummy = torch.randn(1, 3, h, w)
    kwargs = dict(
        input_names=["images"], output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=args.opset,
    )
    try:
        # Newer torch defaults to the dynamo exporter; the legacy TorchScript
        # exporter is what YOLOX's graph was written against.
        torch.onnx.export(model, dummy, args.output, dynamo=False, **kwargs)
    except TypeError:  # older torch: no `dynamo` kwarg, legacy exporter is the only one
        torch.onnx.export(model, dummy, args.output, **kwargs)
    print(f"wrote {args.output}: input (batch, 3, {h}, {w}), dynamic batch, undecoded output")


if __name__ == "__main__":
    main()
