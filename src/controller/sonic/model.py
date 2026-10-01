from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch


class SonicModel:
    """One SONIC decoder invocation for the whole environment batch."""

    def __init__(
        self,
        path: Path,
        batch_size: int,
        input_dim: int,
        device: torch.device,
        cuda_stream: torch.cuda.Stream | None = None,
    ):
        providers: list[str | tuple[str, dict[str, str]]] = ["CPUExecutionProvider"]
        device_id = 0
        if device.type == "cuda":
            device_id = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            ort.preload_dlls()
            options = {"device_id": str(device_id)}
            if cuda_stream is not None:
                options["user_compute_stream"] = str(cuda_stream.cuda_stream)
            providers.insert(0, ("CUDAExecutionProvider", options))
        model = onnx.load(path)
        for value in (model.graph.input[0], model.graph.output[0]):
            batch = value.type.tensor_type.shape.dim[0]
            batch.ClearField("dim_value")
            batch.dim_param = "batch"
        self.session = ort.InferenceSession(
            model.SerializeToString(), providers=providers
        )
        self.run_options = ort.RunOptions()
        if device.type == "cuda" and cuda_stream is not None:
            # SONIC and simulation enqueue work on the same stream.
            self.run_options.add_run_config_entry(
                "disable_synchronize_execution_providers", "1"
            )
        if (
            device.type == "cuda"
            and "CUDAExecutionProvider" not in self.session.get_providers()
        ):
            raise RuntimeError(
                "SONIC decoder did not initialize with CUDAExecutionProvider"
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input = torch.zeros(batch_size, input_dim, device=device)
        self.output = torch.empty(batch_size, 29, device=device)
        self.binding = self.session.io_binding()
        self.binding.bind_input(
            self.input_name,
            device.type,
            device_id,
            np.float32,
            self.input.shape,
            self.input.data_ptr(),
        )
        self.binding.bind_output(
            self.output_name,
            device.type,
            device_id,
            np.float32,
            self.output.shape,
            self.output.data_ptr(),
        )

    def run(self) -> torch.Tensor:
        self.session.run_with_iobinding(self.binding, self.run_options)
        return self.output
