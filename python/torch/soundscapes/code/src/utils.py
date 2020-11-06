import librosa
import numpy as np
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# mixer = torchaudio.transforms.DownmixMono()
from config import SAMPLE_RATE, WINDOW

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])


