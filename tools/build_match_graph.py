#!/usr/bin/env python3
"""Build data/models/xfeat_match.onnx: the GPU mutual-NN descriptor matcher.

A tiny ONNX graph (MatMul + ArgMax/ReduceMax on both axes) that matches
normalised XFeat descriptors on the GPU: for A (nq,64) and B (64,nt) it
returns the best train index per query and the best query index per train,
plus the similarity values. `onnx` is only needed to (re)build the file:

    uv run --with onnx python tools/build_match_graph.py

The resulting file is tiny (a few hundred bytes) and is what
tools/download_models.py can also fetch.
"""

from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "data", "models", "xfeat_match.onnx")


def main() -> None:
    a = helper.make_tensor_value_info("A", TensorProto.FLOAT16, ["nq", 64])
    b = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [64, "nt"])
    fwd_val = helper.make_tensor_value_info("fwd_val", TensorProto.FLOAT16, ["nq"])
    fwd_idx = helper.make_tensor_value_info("fwd_idx", TensorProto.INT64, ["nq"])
    bwd_val = helper.make_tensor_value_info("bwd_val", TensorProto.FLOAT16, ["nt"])
    bwd_idx = helper.make_tensor_value_info("bwd_idx", TensorProto.INT64, ["nt"])
    nodes = [
        helper.make_node("MatMul", ["A", "B"], ["sim"], name="matmul"),
        helper.make_node("ReduceMax", ["sim"], ["fwd_val"], axes=[1], keepdims=0, name="fwd_max"),
        helper.make_node("ArgMax", ["sim"], ["fwd_idx"], axis=1, keepdims=0, name="fwd_arg"),
        helper.make_node("Transpose", ["sim"], ["simT"], perm=[1, 0], name="tr"),
        helper.make_node("ReduceMax", ["simT"], ["bwd_val"], axes=[1], keepdims=0, name="bwd_max"),
        helper.make_node("ArgMax", ["simT"], ["bwd_idx"], axis=1, keepdims=0, name="bwd_arg"),
    ]
    graph = helper.make_graph(nodes, "xfeat_match", [a, b], [fwd_val, fwd_idx, bwd_val, bwd_idx])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    model.producer_name = "wardogs-autopilot"
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    onnx.save(model, TARGET)
    print(f"wrote {TARGET} ({os.path.getsize(TARGET)} bytes)")


if __name__ == "__main__":
    main()
