from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg

@parse_args('v', 'v', 'v', 'i', 'i', 'i')
def fake_quantize_per_channel_affine(g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128, 127] for qint8, got [{}, {}]".format(quant_min, quant_max))
    zero_point_dtype = torch.int8 if quant_min == -128 else torch.uint8

    # fake_quantize_per_channel_affine requires zero_point to be long but ONNX requires uint8 or int8.
    # Get the tensor in the node to convert type
    zero_point = zero_point.node()['value'].to(zero_point_dtype).data
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis),
        scale, zero_point, axis_i=axis)
