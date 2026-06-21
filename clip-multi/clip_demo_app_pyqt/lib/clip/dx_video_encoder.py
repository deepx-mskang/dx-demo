import numpy as np
import torch

from dx_engine import InferenceEngine


class DXVideoEncoder:
    def __init__(self, model_path: str):
        self.ie = InferenceEngine(model_path)
        self.output_dim = 768

    def run_async(self, x, args):
        x = self._prepare_input(x)
        return self.ie.run_async([x], args)

    def wait(self, request_id):
        output = self.ie.Wait(request_id)[0]
        return torch.from_numpy(np.array(output, dtype=np.float32))

    def run(self, x):
        x = self._prepare_input(x)
        output = self.ie.run([x])[0]
        return torch.from_numpy(np.array(output, dtype=np.float32))

    @staticmethod
    def _prepare_input(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        return np.ascontiguousarray(x, dtype=np.float32)
