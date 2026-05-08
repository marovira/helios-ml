import pathlib

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import packaging.version as pv
import torch
import torch.onnx
from torch import nn


def export_to_onnx(
    net: nn.Module,
    net_args: torch.Tensor | tuple[torch.Tensor, ...],
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

    .. note::
        Starting in torch 2.9.1, ``torch.onnx.export`` defaults to ``dynamo=True``. If
        dynamo is used (whether by default in torch >= 2.9.1 or by explicitly passing
        ``dynamo=True`` in ``kwargs``), then the file path is omitted from the arguments
        to ``torch.onnx.export``, using instead ``program.save()`` to write the file. If
        dynamo is not used, then it is passed on to ``torch.onnx.export``.

    Args:
        net: the network to convert.
        net_args: the input tensor(s) for tracing.
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
        out = net(*net_args) if isinstance(net_args, tuple) else net(net_args)

    _dynamo_default = pv.Version(torch.__version__) >= pv.Version("2.9.1")
    is_dynamo = kwargs.get("dynamo", _dynamo_default)

    if is_dynamo:
        args = net_args if isinstance(net_args, tuple) else (net_args,)
        program = torch.onnx.export(net, args, **kwargs)
        assert program is not None
        program.save(str(out_path))
    else:
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

        args_seq = net_args if isinstance(net_args, tuple) else (net_args,)
        ort_inputs = {
            inp.name: to_numpy(t)
            for inp, t in zip(ort_session.get_inputs(), args_seq, strict=True)
        }
        ort_outs = ort_session.run(None, ort_inputs)
        if isinstance(out, dict):
            outs_seq = tuple(out.values())
        elif isinstance(out, tuple | list):
            outs_seq = tuple(out)
        else:
            outs_seq = (out,)
        try:
            for expected, actual in zip(outs_seq, ort_outs, strict=True):
                np.testing.assert_allclose(
                    to_numpy(expected), actual, rtol=rtol, atol=atol
                )
        except AssertionError as e:
            if not save_on_validation_fail:
                out_path.unlink(missing_ok=True)
            raise e
