import librosa
import numpy as np
import torch
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

# mixer = torchaudio.transforms.DownmixMono()
from config import DEF_FREQ, SAMPLE_RATE, WINDOW
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

CUT_SIZE = WINDOW * DEF_FREQ


# Augmentation
def process_file(file_path, freq=DEF_FREQ, train=True):
    # sound = torchaudio.load(file_path, out=None, normalization=True)
    sound, r = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    sound = librosa.util.normalize(sound, axis=0)
    # load returns a tensor with the sound data and the sampling frequency (44.1kHz for UrbanSound8K)
    # sound_np = sound[0].numpy()
    # sound_np = sound
    # if sound_np.shape[0] == 2:
    #     sound_data = librosa.to_mono(sound_np)
    # else:
    #     sound_data = sound_np[0]
    sound_data = sound
    # if train:
    if train:
        sound_data = augment(samples=sound_data, sample_rate=SAMPLE_RATE)
    # sound_data = mixer(augmented)
    # downsample the audio to ~8kHz
    sound_data = torch.from_numpy(sound_data.reshape((sound_data.shape[0], 1)))
    temp_data = torch.zeros([CUT_SIZE, 1])  # temp_data accounts for audio clips that are too short
    if sound_data.numel() <= CUT_SIZE:
        temp_data[:sound_data.numel()] = sound_data[:]
    else:
        rand_index = np.random.randint(0, sound_data.shape[0] - CUT_SIZE)
        temp_data[:] = sound_data[rand_index:rand_index + CUT_SIZE]

    sound_data = temp_data
    sound_formatted = torch.zeros([WINDOW, 1])
    # sound_formatted[:WINDOW] = sound_data[::5]  # take every fifth sample of sound_data
    sound_formatted[:WINDOW] = sound_data[::freq]
    sound_formatted = sound_formatted.permute(1, 0)
    return sound_formatted


def process_sound(sound_data, freq=DEF_FREQ, train=True):
    if train:
        sound_data = augment(samples=sound_data, sample_rate=SAMPLE_RATE)
    # sound_data = mixer(augmented)
    # downsample the audio to ~8kHz
    sound_data = torch.from_numpy(sound_data.reshape((sound_data.shape[0], 1)))
    temp_data = torch.zeros([CUT_SIZE, 1])  # temp_data accounts for audio clips that are too short
    if sound_data.numel() <= CUT_SIZE:
        temp_data[:sound_data.numel()] = sound_data[:]
    else:
        rand_index = np.random.randint(0, high=(sound_data.numel() - CUT_SIZE))
        temp_data[:] = sound_data[rand_index:rand_index + CUT_SIZE]

    sound_data = temp_data
    sound_formatted = torch.zeros([WINDOW, 1])
    # sound_formatted[:WINDOW] = sound_data[::5]  # take every fifth sample of sound_data
    sound_formatted[:WINDOW] = sound_data[::freq]
    sound_formatted = sound_formatted.permute(1, 0)
    return sound_formatted
