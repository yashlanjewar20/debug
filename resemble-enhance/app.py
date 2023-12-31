import gradio as gr
import torch
import torchaudio
import os
import time

from resemble_enhance.enhancer.inference import denoise, enhance


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
os.environ['GRADIO_TEMP_DIR'] = '/home/yash/temp'

def _fn(path, solver, nfe, tau, denoising, run_mode = "fp_16"):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)
    
    
    dwav = dwav.to(device).to(torch.float32)
    
    start_time = time.time()

    # wav1, new_sr = denoise(dwav, sr, device, run_mode)
    wav2, new_sr = enhance(run_mode, dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    end_time= time.time()
    
    duration = end_time - start_time
    #print(f"Processing time: {duration} seconds")

    
    # wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()

    # return (new_sr, wav1), (new_sr, wav2)
    return (new_sr, wav2)

def main():
    inputs: list = [
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(choices=["Midpoint", "RK4", "Euler"], value="Midpoint", label="CFM ODE Solver"),
        gr.Slider(minimum=1, maximum=128, value=64, step=1, label="CFM Number of Function Evaluations"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label="CFM Prior Temperature"),
        gr.Checkbox(value=False, label="Denoise Before Enhancement"),
        gr.Radio(["fp_16", "fp_32"] , label = "fp", default="")
    ]

    outputs: list = [
        gr.Audio(label="Output Denoised Audio"),
        gr.Audio(label="Output Enhanced Audio"),
    ]

    interface = gr.Interface(
        fn=_fn,
        title="Resemble Enhance",
        description="AI-driven audio enhancement for your audio files, powered by Resemble AI.",
        inputs=inputs,
        outputs=outputs,
    )

    interface.launch()


if __name__ == "__main__":
    main()
