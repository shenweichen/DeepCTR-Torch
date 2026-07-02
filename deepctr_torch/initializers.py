# -*- coding:utf-8 -*-
"""Portable, semantic-name-based model initialization."""

import hashlib
import math

import numpy as np
import torch


def _semantic_seed(seed, semantic):
    payload = "{}:{}".format(seed, semantic).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _normal(semantic, shape, seed, stddev):
    random = np.random.Generator(np.random.PCG64(_semantic_seed(seed, semantic)))
    return random.normal(0.0, stddev, size=shape).astype("float32")


def _glorot_normal(semantic, shape, seed):
    fan_in, fan_out = shape[0], shape[1]
    return _normal(
        semantic,
        shape,
        seed,
        math.sqrt(2.0 / float(fan_in + fan_out)),
    )


def _deepfm_semantic(name, shape):
    if name.startswith("embedding_dict.") and name.endswith(".weight"):
        feature = name[len("embedding_dict."):-len(".weight")]
        return "embedding:" + feature, shape, False, "embedding"
    prefix = "linear_model.embedding_dict."
    if name.startswith(prefix) and name.endswith(".weight"):
        feature = name[len(prefix):-len(".weight")]
        return "linear_embedding:" + feature, shape, False, "embedding"
    if name == "linear_model.weight":
        return "linear_dense", shape, False, "embedding"
    if name.startswith("dnn.linears."):
        parts = name.split(".")
        index, kind = parts[2], parts[3]
        semantic = "dnn_{}:{}".format("kernel" if kind == "weight" else "bias", index)
        canonical_shape = (shape[1], shape[0]) if kind == "weight" else shape
        return semantic, canonical_shape, kind == "weight", "glorot" if kind == "weight" else "zero"
    if name == "dnn_linear.weight":
        return "dnn_output", (shape[1], shape[0]), True, "glorot"
    if name == "out.bias":
        return "output_bias", shape, False, "zero"
    raise ValueError("unsupported DeepFM parameter for portable initialization: {}".format(name))


def initialize_deepfm_parameters(model, seed=1024, init_std=0.0001):
    """Initialize DeepFM parameters identically across supported frameworks."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            semantic, canonical_shape, transpose, distribution = _deepfm_semantic(
                name, tuple(parameter.shape)
            )
            if distribution == "embedding":
                value = _normal(semantic, canonical_shape, seed, init_std)
            elif distribution == "glorot":
                value = _glorot_normal(semantic, canonical_shape, seed)
            else:
                value = np.zeros(canonical_shape, dtype="float32")
            if transpose:
                value = value.T
            parameter.copy_(torch.as_tensor(
                value, dtype=parameter.dtype, device=parameter.device
            ))


def export_deepfm_parameters(model):
    """Return canonical NumPy DeepFM parameters for framework transfer."""
    values = {}
    for name, parameter in model.named_parameters():
        semantic, _, transpose, _ = _deepfm_semantic(
            name, tuple(parameter.shape)
        )
        value = parameter.detach().cpu().numpy()
        values[semantic] = value.T.copy() if transpose else value.copy()
    return values


def load_deepfm_parameters(model, parameters):
    """Load canonical NumPy DeepFM parameters from another framework."""
    expected = {
        _deepfm_semantic(name, tuple(parameter.shape))[0]
        for name, parameter in model.named_parameters()
    }
    if set(parameters) != expected:
        raise ValueError(
            "portable parameter coverage mismatch; missing={} unknown={}".format(
                sorted(expected - set(parameters)),
                sorted(set(parameters) - expected),
            )
        )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            semantic, _, transpose, _ = _deepfm_semantic(
                name, tuple(parameter.shape)
            )
            value = np.asarray(parameters[semantic])
            if transpose:
                value = value.T
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError("shape mismatch for {}".format(semantic))
            parameter.copy_(torch.as_tensor(
                value, dtype=parameter.dtype, device=parameter.device
            ))
    return model
