import pathlib

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import torch
import torch.onnx
from torch import nn


def export_to_onnx(
    net: nn.Module,
    net_args: torch.Tensor,
    out_path: pathlib.Path,
    validate_output: bool = False,
    save_on_validation_fail: bool = True,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    **kwargs,
) -> None:
    """
    Export the given network to ONNX format.

    By default, the resulting onnx network will be validated through ONNX to ensure it's
    valid. If you wish to validate the traced outputs to ensure they're the same, set
    ``validate_output`` to true and change ``rtol``/``atol`` as needed.

    .. warning::
        This function assumes that the given network takes a single tensor as input and
        produces a single tensor as output.

    Args:
        net: the network to convert.
        net_args: the input tensor for tracing.
        out_path: the path to save the exported network to.
        validate_output: if true, validation is performed to ensure correctness. Defaults
            to false.
        save_on_validation_fail: if true, the ONNX network is saved regardless of whether
            validation succeeds. If false, the ONNX network is deleted. Defaults to true.
        rtol: relative tolerance threshold. Defaults to ``1e-3``.
        atol: absolute tolerance threshold. Defaults to ``1e-5``.
        kwargs: additional keyword arguments to ``torch.onnx.export``.
    """
    net.eval()
    with torch.no_grad():
        out = net(net_args)

    torch.onnx.export(net, net_args, out_path, **kwargs)  # type: ignore[arg-type]

    onnx.checker.check_model(out_path)

    if validate_output:
        ort_session = onnxruntime.InferenceSession(
            out_path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor: torch.Tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            return tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(net_args)}
        ort_outs = ort_session.run(None, ort_inputs)
        try:
            np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=rtol, atol=atol)
        except AssertionError as e:
            if not save_on_validation_fail:
                out_path.unlink(missing_ok=True)
            raise e
