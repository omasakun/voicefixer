# %%

from typing import Any

import gradio as gr
import numpy as np
import torch
from numpy.typing import NDArray
from torchaudio.functional import resample

from voicefixer import VoiceFixer

NPArray = NDArray[Any]

# Sample rate for VoiceFixer
fixer_sr = 44100

voice_fixer = VoiceFixer()


def fix_audio_dtype(audio: NPArray):
    if audio.dtype == np.int16:
        return audio / 32768.0
    if audio.dtype == np.int32:
        return audio / 2147483648.0
    if audio.dtype == np.float16 or audio.dtype == np.float32 or audio.dtype == np.float64:
        return audio
    raise ValueError("Unsupported dtype")


def infer(source: tuple[int, NPArray], mode: int, use_gpu: bool, volume: float):
    sr, audio = source
    use_gpu = use_gpu and torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"

    audio = fix_audio_dtype(audio)

    audio = torch.from_numpy(audio).to(device)
    audio, sr = resample(audio, sr, fixer_sr), fixer_sr
    audio = audio.cpu().numpy() * 10**(volume / 20)

    # Inference
    fixed_audio = voice_fixer.restore_inmem(audio, mode=mode, cuda=use_gpu)

    fixed_audio = fixed_audio.squeeze()
    return (sr, fixed_audio)


iface = gr.Interface(
    fn=infer,
    inputs=[
        gr.Audio(label="Audio to process", type="numpy"),
        gr.Radio([0, 1, 2], value=0, label="Voice fixer modes (0: original mode, 1: Add preprocessing module, 2: Train mode)"),
        gr.Checkbox(value=True, label="Turn on GPU (if available)"),
        gr.Slider(-40, 40, 0, step=0.1, label="Volume")
    ],
    outputs=[
        gr.Audio(label="Processed audio", type="numpy"),
    ],
)

iface.launch()
