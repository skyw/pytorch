from __future__ import absolute_import, division, print_function, unicode_literals

from torch.onnx.symbolic_helper import parse_args, cast_pytorch_to_onnx

@parse_args('v', 'v', 'v', 'i', 'i', 'i')
def fake_quantize_per_channel_affine(g, inputs, scale, zero_point, axis, quant_min=-128, quant_max=127):
    if quant_min not in [0, -128] or quant_max not in [127, 255]:
        raise RuntimeError(
            "ONNX defines [0, 255] for quint8 and [-128, 127] for qint8, got [{}, {}]".format(quant_min, quant_max))

    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=cast_pytorch_to_onnx['Byte'])
    else:
        zero_point = g.op("Cast", zero_point, to_i=cast_pytorch_to_onnx['Char'])
    return g.op(
        "DequantizeLinear",
        g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis),
        scale, zero_point, axis_i=axis)
