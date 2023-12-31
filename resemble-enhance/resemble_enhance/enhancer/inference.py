import logging
from functools import cache

import torch
import time
from ..inference import inference
from .download import download
from .train import Enhancer, HParams

logger = logging.getLogger(__name__)


@cache
def load_enhancer(run_dir, device, run_mode):
    if run_dir is None:
        run_dir = download()
    hp = HParams.load(run_dir)
    enhancer = Enhancer(run_mode, hp)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu")["module"]
    enhancer.load_state_dict(state_dict)
    enhancer.eval()
    enhancer.to(device)
    
    if run_mode == "fp_16":
        print("fp_16 here")
        enhancer.half() 
        
    return enhancer


@torch.inference_mode()
def denoise(dwav, sr, device, run_mode, run_dir=None):
    if run_mode == "fp_16":
        dwav = dwav.to(device).half()
        print("fp_16 here")
    enhancer = load_enhancer(run_dir, device, run_mode)
    return inference(model=enhancer.denoiser, dwav=dwav, sr=sr, device=device)


@torch.inference_mode()
def enhance(run_mode, dwav, sr, device, nfe=32, solver="midpoint", lambd=0.5, tau=0.5, run_dir=None):
    assert 0 < nfe <= 128, f"nfe must be in (0, 128], got {nfe}"
    assert solver in ("midpoint", "rk4", "euler"), f"solver must be in ('midpoint', 'rk4', 'euler'), got {solver}"
    assert 0 <= lambd <= 1, f"lambd must be in [0, 1], got {lambd}"
    assert 0 <= tau <= 1, f"tau must be in [0, 1], got {tau}"
    
    dwav = dwav.to(device).half()
    enhancer = load_enhancer(run_dir, device, run_mode)
    enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    
    for i in range(10):
        st=time.time()
        out = inference(model=enhancer, dwav=dwav, sr=sr, device=device)
        print(time.time()-st)
    return out
