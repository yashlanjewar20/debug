import logging
from functools import cache

import torch

from ..inference import inference
from .train import Denoiser, HParams

logger = logging.getLogger(__name__)


@cache
def load_denoiser(run_dir, device, run_mode):
    if run_dir is None:
        return Denoiser(run_mode, HParams())
    hp = HParams.load(run_dir)
    denoiser = Denoiser(run_mode, hp)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu")["module"]
    denoiser.load_state_dict(state_dict)
    denoiser.eval()
    denoiser.to(device)
    if run_mode == "fp_16":
        denoiser.half()
    return denoiser


@torch.inference_mode()
def denoise(dwav, sr, run_dir, device, run_mode):
    denoiser = load_denoiser(run_dir, device, run_mode)
    return inference(model=denoiser, dwav=dwav, sr=sr, device=device)
